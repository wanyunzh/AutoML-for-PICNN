import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
import os
import random
from scipy.interpolate import interp1d
import tikzplotlib
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset,FixGeoDataset,VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor
# from modelself import USCNN, USCNNSep
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
import nni.retiarii.nn.pytorch as nn
from hpo_utils import *
import Ofpp
import nni
from loss_operation import *


class USCNNSep(nn.Module):
	def __init__(self,h,nx,ny,nVarIn=1,nVarOut=1,initWay=None,k=5,s=1,p=2):
		super(USCNNSep, self).__init__()
		"""
		Extract basic information
		"""
		self.initWay=initWay
		self.nVarIn=nVarIn
		self.nVarOut=nVarOut
		self.k=k
		self.s=1
		self.p=2
		self.deltaX=h
		self.nx=nx
		self.ny=ny
		"""
		Define net
		"""
		W1=16
		W2=32
		self.relu=nn.ReLU()
		self.US=nn.Upsample(size=[self.ny-2,self.nx-2],mode='bicubic')
		self.conv1=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
		self.conv2=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
		self.conv3=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
		self.conv4=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
		self.pixel_shuffle1 = nn.PixelShuffle(1)
		self.conv11=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
		self.conv22=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
		self.conv33=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
		self.conv44=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)
		self.pixel_shuffle11 = nn.PixelShuffle(1)
		self.conv111=nn.Conv2d(self.nVarIn,W1,kernel_size=k, stride=s, padding=p)
		self.conv222=nn.Conv2d(W1,W2,kernel_size=k, stride=s, padding=p)
		self.conv333=nn.Conv2d(W2,W1,kernel_size=k, stride=s, padding=p)
		self.conv444=nn.Conv2d(W1,self.nVarOut,kernel_size=k, stride=s, padding=p)

	def forward(self, x):
		x=self.US(x)
		x1=self.relu(self.conv1(x))
		x1=self.relu(self.conv2(x1))
		x1=self.relu(self.conv3(x1))
		x1=self.pixel_shuffle1(self.conv4(x1))

		x2=self.relu(self.conv11(x))
		x2=self.relu(self.conv22(x2))
		x2=self.relu(self.conv33(x2))
		x2=self.pixel_shuffle11(self.conv44(x2))

		x3=self.relu(self.conv111(x))
		x3=self.relu(self.conv222(x3))
		x3=self.relu(self.conv333(x3))
		x3=self.conv444(x3)
		return  torch.cat([x1,x2,x3],axis=1)

# class USCNNSep(nn.Module):
#     def __init__(self, params, nVarIn, nVarOut):
#         super().__init__()
#         """
#         Extract basic information
#         """
#         self.relu = nn.ReLU()
#         self.params = params
#         self.nVarIn=nVarIn
#         self.nVarOut=nVarOut
#         self.conv1 = nn.Conv2d(self.nVarIn, params['channel1'], kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(params['channel1'], params['channel2'], kernel_size=5, stride=1, padding=2)
#         self.conv3 = nn.Conv2d(params['channel2'], params['channel3'], kernel_size=5, stride=1, padding=2)
#         self.conv4 = nn.Conv2d(params['channel3'], params['channel2'], kernel_size=5, stride=1, padding=2)
#         self.conv5 = nn.Conv2d(params['channel2'], params['channel1'], kernel_size=5, stride=1, padding=2)
#         self.conv6 = nn.Conv2d(params['channel1'], self.nVarOut, kernel_size=5, stride=1, padding=2)
#
#
#
#     def forward(self, x):
#
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))
#         x = self.relu(self.conv5(x))
#         x = self.conv6(x)
#         return x

def traintest_case3():
        seed = 123
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        params = {
            'constraint': 2,
            'UNARY_OPS': 1,
            'WEIGHT_INIT': 0,
            'WEIGHT_OPS': 3,
            'gradient': 0,
            'kernel': 2,
        }
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        print(device)

        params1 = {
            'channel1': 16,
            'channel2': 32,
            'channel3': 64,
            'channel4': 16
        }


        nxOF = 50
        nyOF = 50
        nx = nxOF
        ny = nyOF
        h = 0.01
        NvarInput = 2
        NvarOutput = 3
        model = USCNNSep(params1, NvarInput, NvarOutput).to(device)
        scalarList = [-0.1, 0.0, 0.1]
        SolutionList = []
        MeshList = []
        for scalar in scalarList:
            OFcaseName_ = './TemplateCase' + str(scalar)
            # nx = nxOF + 2;
            # ny = nyOF + 2;
            nx = nxOF
            ny = nyOF
            R = 0.5
            L = 0
            l = 0.5
            h = 0.01
            idx = np.asarray(range(1, nx - 1, 1))
            idy = np.asarray(range(1, ny - 1, 1))
            leftY = np.linspace(-l / 2 - L / 2, l / 2 + L / 2, ny)
            rightY = np.linspace(-l / 2 - L / 2, l / 2 + L / 2, ny)
            leftX = []
            rightX = []
            for i in leftY:
                if i > -l / 2 and i < l / 2:
                    leftX.append(+np.cos(2 * np.pi * i) * scalar - R)
                    rightX.append(-np.cos(2 * np.pi * i) * scalar + R)
                else:  # 顶点值直接就知道了
                    leftX.append(-R);
                    rightX.append(R)
            leftX = np.asarray(leftX)
            rightX = np.asarray(rightX)
            lowX = np.linspace(-R, R, nx);
            lowY = lowX * 0 - l / 2 - L / 2
            upX = lowX;
            upY = lowY + l + L
            myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                               lowX, lowY, upX, upY, h, False, True, './Mesh' + str(scalar) + '.pdf')

            MeshList.append(myMesh)
            OFPic = convertOFMeshToImage(OFcaseName_ + '/3200/C', [OFcaseName_ + '/3200/U', OFcaseName_ + '/3200/p'],
                                         [0, 1, 0, 1], 0.0, False)
            OFU = OFPic[:, :, 2]
            OFV = OFPic[:, :, 3]
            OFP = OFPic[:, :, 4]
            Vmag_True = np.sqrt(OFU ** 2 + OFV ** 2)
            SolutionList.append(OFPic[:, :, 2:])
        batchSize = len(scalarList)

        nEpochs = 20000
        lr = 0.001
        criterion = nn.MSELoss()

        padSingleSide = 1
        udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
        train_set = VaryGeoDataset_PairedSolution(MeshList, SolutionList)
        training_data_loader = DataLoader(dataset=train_set,
                                          batch_size=batchSize)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                        steps_per_epoch=len(training_data_loader),
                                                        epochs=nEpochs, div_factor=2)
        Res_list = []
        Error = []
        id_list = []
        step = 0
        value = 10
        XRes = []
        YRes = []
        MRes = []
        EU = []
        EV = []
        EP = []
        EVmag = []
        for epoch in range(0, nEpochs):
            print('epoch', epoch)
            Res = 0
            yRes = 0
            mRes = 0
            eU = 0
            eV = 0
            eP = 0
            eVmag = 0

            for iteration, batch in enumerate(training_data_loader):
                [JJInv, coord, xi,
                 eta, J, Jinv,
                 dxdxi, dydxi,
                 dxdeta, dydeta,
                 Utrue, Vtrue, Ptrue] = \
                    to4DTensor(batch,device)
                solutionTruth = torch.cat([Utrue, Vtrue, Ptrue], axis=1)
                optimizer.zero_grad()
                output = model(coord)
                if (epoch == 0) and (iteration) == 0:

                    if params['WEIGHT_INIT'] == 1:
                        init_weight = torch.ones_like(output)
                    else:
                        init_weight = torch.zeros_like(output)
                    WEIGHT_OPS1 = {
                        0: P_OHEM1(init_weight),
                        1: Loss_Adaptive1(init_weight),
                        2: Max1(init_weight),
                        3: One(init_weight),
                    }

                    WEIGHT_OPS2 = {
                        0: P_OHEM2(init_weight),
                        1: Loss_Adaptive2(init_weight),
                        2: Max2(init_weight),
                        3: One(init_weight),
                    }

                    WEIGHT_OPS3 = {
                        0: P_OHEM3(init_weight),
                        1: Loss_Adaptive3(init_weight),
                        2: Max3(init_weight),
                        3: One(init_weight),
                    }

                outputU0 = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                      output.shape[2],
                                                      output.shape[3])
                outputV0 = output[:, 1, :, :].reshape(output.shape[0], 1,
                                                      output.shape[2],
                                                      output.shape[3])
                outputP0 = output[:, 2, :, :].reshape(output.shape[0], 1,
                                                      output.shape[2],
                                                      output.shape[3])
                output_pad = output.clone()
                outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                         output_pad.shape[2],
                                                         output_pad.shape[3])
                outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1,
                                                         output_pad.shape[2],
                                                         output_pad.shape[3])
                outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1,
                                                         output_pad.shape[2],
                                                         output_pad.shape[3])

                loss_boundary = 0
                for j in range(batchSize):
                    bc1 = outputU0[j, 0, :padSingleSide, padSingleSide:-padSingleSide] - 0
                    bc2 = outputU0[j, 0, :, -padSingleSide:] - 0
                    bc3 = outputU0[j, 0, :, 0:padSingleSide] - 0
                    bc11 = outputV0[j, 0, :padSingleSide, padSingleSide:-padSingleSide] - 0.4
                    bc22 = outputV0[j, 0, padSingleSide:, -padSingleSide:] - 0
                    bc33 = outputV0[j, 0, padSingleSide:, 0:padSingleSide] - 0
                    bc44 = outputV0[j, 0, 0, 0] - 0.2
                    bc55 = outputV0[j, 0, 0, -1] - 0.2
                    bc111 = outputP[j, 0, -padSingleSide:, :] - 0
                    loss_boundary = loss_boundary + 1 * UNARY_OPS[params['UNARY_OPS']](bc1).sum() + 1 * UNARY_OPS[
                        params['UNARY_OPS']](bc2).sum() + 1 * UNARY_OPS[params['UNARY_OPS']](bc3).sum() + \
                                    1 * UNARY_OPS[params['UNARY_OPS']](bc11).sum() + 1 * UNARY_OPS[
                                        params['UNARY_OPS']](bc22).sum() + UNARY_OPS[params['UNARY_OPS']](
                        bc33).sum() + \
                                    1 * UNARY_OPS[params['UNARY_OPS']](bc44) + 1 * UNARY_OPS[params['UNARY_OPS']](
                        bc55) + \
                                    1 * UNARY_OPS[params['UNARY_OPS']](bc111).sum()
                loss_boundary = loss_boundary / (batchSize * (2 * bc1.shape[1] + 5 * bc2.shape[0]))

                for j in range(batchSize):
                    outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -2,
                                                                                   padSingleSide:-padSingleSide]
                    outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
                    outputU[j, 0, :, -padSingleSide:] = 0
                    outputU[j, 0, :, 0:padSingleSide] = 0
                    outputU[j, 0, 0, 0] = 0.5 * (outputU[j, 0, 0, 1] + outputU[j, 0, 1, 0])
                    outputU[j, 0, 0, -1] = 0.5 * (outputU[j, 0, 0, -2] + outputU[j, 0, 1, -1])
                    outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -2,
                                                                                   padSingleSide:-padSingleSide]
                    outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.4
                    outputV[j, 0, padSingleSide:, -padSingleSide:] = 0
                    outputV[j, 0, padSingleSide:, 0:padSingleSide] = 0
                    outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
                    outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
                    outputP[j, 0, -padSingleSide:, :] = 0
                    outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 1,
                                                                                  padSingleSide:-padSingleSide]
                    outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2,
                                                                                   padSingleSide:-padSingleSide,
                                                                                   -2:-1]
                    outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2,
                                                                                   padSingleSide:-padSingleSide,
                                                                                   1:2]
                    outputP[j, 0, 0, 0] = 0.5 * (outputP[j, 0, 0, 1] + outputP[j, 0, 1, 0])
                    outputP[j, 0, 0, -1] = 0.5 * (outputP[j, 0, 0, -2] + outputP[j, 0, 1, -1])
                kernel = params['kernel']
                diff_filter = Filter((output.shape[2], outputV.shape[3]), filter_size=kernel, device=device)
                if params['constraint'] == 2:
                    loss_boundary = 0

                if params['constraint'] == 0:
                    dudx, dudy = pde_out(outputU0, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    d2udx2, _ = pde_out(dudx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    _, d2udy2 = pde_out(dudy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    dvdx, dvdy = pde_out(outputV0, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    d2vdx2, _ = pde_out(dvdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    _, d2vdy2 = pde_out(dvdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    dpdx, dpdy = pde_out(outputP0, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    d2pdx2, _ = pde_out(dpdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    _, d2pdy2 = pde_out(dpdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    continuity, Xresidual, Yresidual = pde_residue(dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2,
                                                                   d2udy2,
                                                                   d2vdx2, d2vdy2, outputU, outputV)
                else:
                    dudx, dudy = pde_out(outputU, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    d2udx2, _ = pde_out(dudx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    _, d2udy2 = pde_out(dudy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    dvdx, dvdy = pde_out(outputV, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    d2vdx2, _ = pde_out(dvdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    _, d2vdy2 = pde_out(dvdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    dpdx, dpdy = pde_out(outputP, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    d2pdx2, _ = pde_out(dpdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    _, d2pdy2 = pde_out(dpdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    continuity, Xresidual, Yresidual = pde_residue(dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2,
                                                                   d2udy2,
                                                                   d2vdx2, d2vdy2, outputU, outputV)

                difference1 = continuity - torch.zeros_like(continuity)
                difference2 = Xresidual - torch.zeros_like(Xresidual)
                difference3 = Yresidual - torch.zeros_like(Yresidual)

                post_difference1 = UNARY_OPS[params['UNARY_OPS']](difference1)
                post_difference2 = UNARY_OPS[params['UNARY_OPS']](difference2)
                post_difference3 = UNARY_OPS[params['UNARY_OPS']](difference3)
                weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_difference1, epoch)
                loss_search1 = torch.mean(torch.abs(post_difference1 * weight_search1))
                weight_search2 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_difference2, epoch)
                loss_search2 = torch.mean(torch.abs(post_difference2 * weight_search1))
                weight_search3 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_difference3, epoch)
                loss_search3 = torch.mean(torch.abs(post_difference3 * weight_search3))
                loss_search = loss_search1 + loss_search2 + loss_search3

                if params['gradient'] == 1:
                    if epoch >= 2000:

                        dr1dx, dr1dy = pde_out(difference1, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        dr2dx, dr2dy = pde_out(difference2, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        dr3dx, dr3dy = pde_out(difference3, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                        post_gradient1 = UNARY_OPS[params['UNARY_OPS']](dr1dx)
                        post_gradient2 = UNARY_OPS[params['UNARY_OPS']](dr1dy)
                        post_gradient3 = UNARY_OPS[params['UNARY_OPS']](dr2dx)
                        post_gradient4 = UNARY_OPS[params['UNARY_OPS']](dr2dy)
                        post_gradient5 = UNARY_OPS[params['UNARY_OPS']](dr3dx)
                        post_gradient6 = UNARY_OPS[params['UNARY_OPS']](dr3dy)

                        gradient_weight_search1 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient1, epoch)
                        gradient_weight_search2 = WEIGHT_OPS1[params['WEIGHT_OPS']](post_gradient2, epoch)
                        gradient_weight_search3 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient3, epoch)
                        gradient_weight_search4 = WEIGHT_OPS2[params['WEIGHT_OPS']](post_gradient4, epoch)
                        gradient_weight_search5 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient5, epoch)
                        gradient_weight_search6 = WEIGHT_OPS3[params['WEIGHT_OPS']](post_gradient6, epoch)

                        gradient_loss_search1 = torch.mean(torch.abs(post_gradient1 * gradient_weight_search1))
                        gradient_loss_search2 = torch.mean(torch.abs(post_gradient2 * gradient_weight_search2))
                        gradient_loss_search3 = torch.mean(torch.abs(post_gradient3 * gradient_weight_search3))
                        gradient_loss_search4 = torch.mean(torch.abs(post_gradient4 * gradient_weight_search4))
                        gradient_loss_search5 = torch.mean(torch.abs(post_gradient5 * gradient_weight_search5))
                        gradient_loss_search6 = torch.mean(torch.abs(post_gradient6 * gradient_weight_search6))
                        if epoch % 200 == 0:
                            print(gradient_loss_search1)
                            print(gradient_loss_search2)
                            print(gradient_loss_search3)
                            print(gradient_loss_search4)
                            print(gradient_loss_search5)
                            print(gradient_loss_search6)

                    else:
                        gradient_loss_search1 = 0
                        gradient_loss_search2 = 0
                        gradient_loss_search3 = 0
                        gradient_loss_search4 = 0
                        gradient_loss_search5 = 0
                        gradient_loss_search6 = 0
                else:
                    gradient_loss_search1 = 0
                    gradient_loss_search2 = 0
                    gradient_loss_search3 = 0
                    gradient_loss_search4 = 0
                    gradient_loss_search5 = 0
                    gradient_loss_search6 = 0
                loss = loss_search + 10 * loss_boundary + 1 * 1e-4 * (
                        gradient_loss_search1 + gradient_loss_search2 + gradient_loss_search3 + gradient_loss_search4 + gradient_loss_search5 + gradient_loss_search6)

                loss.backward()
                optimizer.step()
                # scheduler.step()
                Res = Res + loss.item()
                eU = eU + torch.sqrt(criterion(Utrue, outputU) / criterion(Utrue, Utrue * 0))
                eV = eV + torch.sqrt(criterion(Vtrue, outputV) / criterion(Vtrue, Vtrue * 0))
                eP = eP + torch.sqrt(criterion(Ptrue, outputP) / criterion(Ptrue, Ptrue * 0))
                Vmag_True = torch.sqrt(Utrue ** 2 + Vtrue ** 2)
                Vmag = torch.sqrt(outputU ** 2 + outputV ** 2)
                eVmag = eVmag + torch.sqrt(criterion(Vmag_True, Vmag) / criterion(Vmag_True, Vmag_True * 0))

            print('Epoch is ', epoch)
            # print("xRes Loss is", (xRes / len(training_data_loader)))
            # print("yRes Loss is", (yRes / len(training_data_loader)))
            # print("mRes Loss is", (mRes / len(training_data_loader)))
            # print("eU Loss is", (eU / len(training_data_loader)))
            print("eVmag Loss is", (eVmag / len(training_data_loader)))
            print("eP Loss is", (eP / len(training_data_loader)))
            # xres=xRes / len(training_data_loader)
            # yres=yRes / len(training_data_loader)
            # mres=mRes / len(training_data_loader)
            eu = eU / len(training_data_loader)
            ev = eV / len(training_data_loader)
            ep = eP / len(training_data_loader)
            evmag = eVmag / len(training_data_loader)
            # XRes.append(xres)
            # YRes.append(yres)
            # MRes.append(mres)
            EU.append(eu)
            EV.append(ev)
            EP.append(ep)
            EVmag.append(evmag)
            if epoch > 5000:
                if ep <= value:
                    value = ep
                    numepoch = epoch
                    print(numepoch)
                    torch.save(model.state_dict(), 'retrain_NS.pth')
            # if epoch == 10000:
            #     # print('value is:',value)
            #     nni.report_intermediate_result(float(value))
        error_finalv = EVmag[numepoch]
        error_final = EP[numepoch]
        error_final = error_final.cpu().detach().numpy()
        error_finalv = error_finalv.cpu().detach().numpy()
        error_final = float(error_final)
        error_finalv=float(error_finalv)
        print('training l2 perror', error_final)
        print('training l2 verror:', error_finalv)
        nni.report_intermediate_result(float(error_final))

        nxOF = 50
        nyOF = 50
        # scalarList = [-0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1]
        scalarList = [-0.075, -0.05, -0.025, 0.025, 0.05, 0.075]
        SolutionList = []
        MeshList = []
        for scalar in scalarList:
            OFcaseName_ = './TemplateCase' + str(scalar)
            nx = nxOF
            ny = nyOF
            R = 0.5
            L = 0
            l = 0.5
            h = 0.01
            idx = np.asarray(range(1, nx - 1, 1))
            idy = np.asarray(range(1, ny - 1, 1))
            leftY = np.linspace(-l / 2 - L / 2, l / 2 + L / 2, ny)
            rightY = np.linspace(-l / 2 - L / 2, l / 2 + L / 2, ny)
            leftX = []
            rightX = []
            for i in leftY:
                if i > -l / 2 and i < l / 2:
                    leftX.append(+np.cos(2 * np.pi * i) * scalar - R)
                    rightX.append(-np.cos(2 * np.pi * i) * scalar + R)
                else:
                    leftX.append(-R)
                    rightX.append(R)
            leftX = np.asarray(leftX)
            rightX = np.asarray(rightX)
            lowX = np.linspace(-R, R, nx)
            lowY = lowX * 0 - l / 2 - L / 2
            upX = lowX
            upY = lowY + l + L
            myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                               lowX, lowY, upX, upY, h, False, True, './Mesh' + str(scalar) + '.pdf')
            MeshList.append(myMesh)
            OFPic = convertOFMeshToImage(OFcaseName_ + '/3200/C', [OFcaseName_ + '/3200/U', OFcaseName_ + '/3200/p'],
                                         [0, 1, 0, 1], 0.0, False)
            OFU = OFPic[:, :, 2]
            OFV = OFPic[:, :, 3]
            OFP = OFPic[:, :, 4]
            SolutionList.append(OFPic[:, :, 2:])
        criterion = nn.MSELoss()
        padSingleSide = 1
        test_set = VaryGeoDataset_PairedSolution(MeshList, SolutionList)
        VelocityMagnitudeErrorRecord = []
        PErrorRecord = []
        model.load_state_dict(torch.load('retrain_NS.pth'))

        for i in range(len(scalarList)):
            [JJInv, coord, xi,
             eta, J, Jinv,
             dxdxi, dydxi,
             dxdeta, dydeta,
             Utrue, Vtrue, Ptrue] = \
                to4DTensor(test_set[i],device)
            solutionTruth = torch.cat([Utrue, Vtrue, Ptrue], axis=1)
            coord = coord.reshape(coord.shape[1], coord.shape[0], coord.shape[2], coord.shape[3])
            model.eval()
            output = model(coord)
            output_pad = output
            outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])

            for j in range(1):
                outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -2,
                                                                               padSingleSide:-padSingleSide]
                outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
                outputU[j, 0, :, -padSingleSide:] = 0
                outputU[j, 0, :, 0:padSingleSide] = 0
                outputU[j, 0, 0, 0] = 0.5 * (outputU[j, 0, 0, 1] + outputU[j, 0, 1, 0])
                outputU[j, 0, 0, -1] = 0.5 * (outputU[j, 0, 0, -2] + outputU[j, 0, 1, -1])
                outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -2,
                                                                               padSingleSide:-padSingleSide]
                outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.4
                outputV[j, 0, padSingleSide:, -padSingleSide:] = 0
                outputV[j, 0, padSingleSide:, 0:padSingleSide] = 0
                outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
                outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
                outputP[j, 0, -padSingleSide:, :] = 0
                outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 1,
                                                                              padSingleSide:-padSingleSide]
                outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2,
                                                                               padSingleSide:-padSingleSide, -2:-1]
                outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2,
                                                                               padSingleSide:-padSingleSide, 1:2]
                outputP[j, 0, 0, 0] = 0.5 * (outputP[j, 0, 0, 1] + outputP[j, 0, 1, 0])
                outputP[j, 0, 0, -1] = 0.5 * (outputP[j, 0, 0, -2] + outputP[j, 0, 1, -1])

            outputU = outputU[0, 0, :, :]
            outputV = outputV[0, 0, :, :]
            outputP=outputP[0,0,:,:]

            Vmag_True = torch.sqrt(Utrue ** 2 + Vtrue ** 2)
            Vmag_ = torch.sqrt(outputU ** 2 + outputV ** 2)

            VelocityMagnitudeErrorRecord.append(
                torch.sqrt(criterion(Vmag_True, Vmag_) / criterion(Vmag_True, Vmag_True * 0)))
            PErrorRecord.append(torch.sqrt(criterion(Ptrue, outputP) / criterion(Ptrue, Ptrue * 0)))
        VErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
        PErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in PErrorRecord])
        # ep_final = EP[numepoch]
        # ep_final = ep_final.cpu().detach().numpy()
        # ep_final = float(ep_final)
        # ev_final = EVmag[numepoch]
        # ev_final = float(ev_final)
        # print('ep_final', ep_final)
        # print('ev_final', ev_final)
        # nni.report_intermediate_result(1 / ep_final)
        Verror_mean = np.mean(VErrorNumpy)
        Perror_mean = np.mean(PErrorNumpy)
        print('Verror:', Verror_mean)
        print('Perror', Perror_mean)
        out=float(1/Perror_mean)
        nni.report_final_result(out)
if __name__ == "__main__":
    traintest_case3()