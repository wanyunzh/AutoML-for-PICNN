import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
from scipy.interpolate import interp1d
import pickle
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor
from model import USCNN
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from evaluate_case2 import traintest_case2
# from model_uscnn import USCNN
################################################################################
from sklearn.metrics import mean_squared_error as calMSE
import nni
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import nni.retiarii.strategy as strategy
import torch
import torch.nn as nn
import torch.nn.init as init
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import os
# from case2_oneshot import enaself_model,spoself_model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)
import Ofpp
import random
torch.manual_seed(123)


def dfdx(f, dydeta, dydxi, Jinv):
    # Equation 13（a） in paper
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx

def dfdy(f, dxdxi, dxdeta, Jinv):
    # Equation 13（b） in paper
    dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 / h
    dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :, 3:]) / 6 / h
    dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :, 0:-3]) / 6 / h
    dfdxi = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)

    dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 / h
    dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:, :]) / 6 / h
    dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 / h
    dfdeta = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
    dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy

class DepthwiseSeparableConv3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3,padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=5,padding=2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv7(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=7,padding=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

@model_wrapper
class USCNN(nn.Module):
    def __init__(self, params,h: float, nx: int, ny: int, nVarIn: int = 2, nVarOut: int = 1, initWay= None,k=5,s=1,p=2):
        super().__init__()
        """
        Extract basic information
        """
        self.params=params
        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut

        self.s = 1
        self.p = 2
        self.deltaX = h
        self.nx = nx
        self.ny = ny
        """
        Define net
        """
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.conv1 = nn.Conv2d(self.nVarIn, params['channel1'], kernel_size=3, stride=1, padding=1)
        # self.conv1 = DepthwiseSeparableConv3(self.nVarIn, params['channel1'])
        # self.conv1 = nn.LayerChoice([
        #     nn.Conv2d(self.nVarIn, 16, kernel_size=5, stride=1, padding=2),
        #     nn.Conv2d(self.nVarIn, 16, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(self.nVarIn, 16, kernel_size=7, stride=1, padding=3),
        #     DepthwiseSeparableConv5(nVarIn, 16),
        #     # DepthwiseSeparableConv3(nVarIn, 16),
        #     # DepthwiseSeparableConv7(nVarIn, 16)
        # ])
        self.conv2 = nn.Conv2d(params['channel1'], params['channel2'], kernel_size=3, stride=1, padding=1)
        # self.conv2 = DepthwiseSeparableConv5(params['channel1'], params['channel2'])
        # self.conv2 = nn.LayerChoice([
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),
        #     DepthwiseSeparableConv5(16, 32),
        # DepthwiseSeparableConv3(16, 32),
        # DepthwiseSeparableConv7(16, 32)
        # ])
        # self.conv1_1 = DepthwiseSeparableConv3(params['channel2'], params['channel2'])
        self.conv1_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv1_2 = nn.MaxPool2d(3, stride=1, padding=1)

        # self.conv1_1 = nn.Sequential( nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()))

        self.conv3 = nn.Conv2d(params['channel2'], params['channel3'], kernel_size=3, stride=1, padding=1)
        # self.conv3 = DepthwiseSeparableConv5(params['channel2'], params['channel1'])
        self.conv4 = nn.Conv2d(params['channel3'], self.nVarOut, kernel_size=3, stride=1, padding=1)
        # self.conv4 = DepthwiseSeparableConv3(params['channel1'], self.nVarOut)

        self.pixel_shuffle = nn.PixelShuffle(1)
        if self.initWay is not None:
            self._initialize_weights()



    def forward(self, x):
        x = self.US(x)  # 19*84变为17*82
        x = self.relu(self.conv1(x))  # channel由2变成了16,大小不变，kernel size=5， padding=2
        x = self.relu(self.conv2(x))# channel由16变成了32，大小不变，kernel size=5， padding=2
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.relu(self.conv3(x))  # channel由32变成了16，大小不变，kernel size=5， padding=2
        x = self.conv4(x)  # channel由16变成了1，大小不变，kernel size=5， padding=2
        x = self.pixel_shuffle(x)  # 不变还是原来的维度
        return x

    def _initialize_weights(self):
        if self.initWay == 'kaiming':
            for i in [0,1,2]:
                init.kaiming_normal_(self.conv1[i].weight, mode='fan_out', nonlinearity='relu')
                init.kaiming_normal_(self.conv2[i].weight, mode='fan_out', nonlinearity='relu')
                init.kaiming_normal_(self.conv3[i].weight, mode='fan_out', nonlinearity='relu')
                init.kaiming_normal_(self.conv4[i].weight,mode='fan_out', nonlinearity='relu')
            # for i in [3,4,5]:
            #     for j in [0,1,2]:
            #         init.kaiming_normal_(self.conv1[i][j].weight, mode='fan_out', nonlinearity='relu')
            #         init.kaiming_normal_(self.conv2[i][j].weight, mode='fan_out', nonlinearity='relu')
            #         init.kaiming_normal_(self.conv3[i][j].weight, mode='fan_out', nonlinearity='relu')
            #         init.kaiming_normal_(self.conv4[i][j].weight, mode='fan_out', nonlinearity='relu')

        elif self.initWay == 'ortho':
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)
        else:
            print('Only Kaiming or Orthogonal initializer can be used!')
            exit()

# @model_wrapper
# class USCNN(nn.Module):
#     def __init__(self, h: float, nx: int, ny: int, nVarIn: int = 2, nVarOut: int = 1, initWay= None,k=5,s=1,p=2):
#         super().__init__()
#         """
#         Extract basic information
#         """
#         self.initWay = initWay
#         self.nVarIn = nVarIn
#         self.nVarOut = nVarOut
#
#         self.s = 1
#         self.p = 2
#         self.deltaX = h
#         self.nx = nx
#         self.ny = ny
#         """
#         Define net
#         """
#         self.relu = nn.ReLU()
#         self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
#         self.conv1=nn.Conv2d(self.nVarIn,16,kernel_size=k, stride=s, padding=p)
#
#         self.conv2=nn.Conv2d(16,32,kernel_size=k, stride=s, padding=p)
#
#
#
#         self.conv_dep1 = nn.Repeat(nn.Sequential(nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
#                                                  nn.ReLU()), (5, 7))
#         self.conv3=nn.Conv2d(32,16,kernel_size=k, stride=s, padding=p)
#
#         self.conv4=nn.Conv2d(16,self.nVarOut,kernel_size=k, stride=s, padding=p)
#
#         self.pixel_shuffle = nn.PixelShuffle(1)
#         if self.initWay is not None:
#             self._initialize_weights()
#         # Specify filter
#
#         self.convdx = nn.Conv2d(1, 1, (5, 5), stride=1, padding=0, bias=None)
#
#
#     def forward(self, x):
#         x = self.US(x)  # 19*84变为17*82
#         x = self.relu(self.conv1(x))  # channel由2变成了16,大小不变，kernel size=5， padding=2
#         x = self.relu(self.conv2(x))  # channel由16变成了32，大小不变，kernel size=5， padding=2
#         x = self.conv_dep1(x)
#         x = self.relu(self.conv3(x))  # channel由32变成了16，大小不变，kernel size=5， padding=2
#         x = self.conv4(x)  # channel由16变成了1，大小不变，kernel size=5， padding=2
#         x = self.pixel_shuffle(x)  # 不变还是原来的维度
#         # x=(self.conv4(x))
#         return x
#
#     def _initialize_weights(self):
#         if self.initWay == 'kaiming':
#             for i in [0,1,2]:
#                 init.kaiming_normal_(self.conv1[i].weight, mode='fan_out', nonlinearity='relu')
#                 init.kaiming_normal_(self.conv2[i].weight, mode='fan_out', nonlinearity='relu')
#                 init.kaiming_normal_(self.conv3[i].weight, mode='fan_out', nonlinearity='relu')
#                 init.kaiming_normal_(self.conv4[i].weight)
#         elif self.initWay == 'ortho':
#             init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
#             init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
#             init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
#             init.orthogonal_(self.conv4.weight)
#         else:
#             print('Only Kaiming or Orthogonal initializer can be used!')
#             exit()

if __name__ == "__main__":
    params = {
        'lr': 0.001,
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }
    import nni
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    h = 0.01
    r = 0.5
    R = 1
    dtheta = 0
    OFBCCoord = Ofpp.parse_boundary_field('TemplateCase/30/C')
    OFLEFTC = OFBCCoord[b'left'][b'value']
    OFRIGHTC = OFBCCoord[b'right'][b'value']
    leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
    leftY = r * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
    rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
    rightY = R * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
    lowX = np.linspace(leftX[0], rightX[0], 49);
    lowY = lowX * 0 + np.sin(dtheta)
    upX = np.linspace(leftX[-1], rightX[-1], 49);
    upY = upX * 0 - np.sin(dtheta)
    ny = len(leftX);
    nx = len(lowX)
    myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                       lowX, lowY, upX, upY, h, True, True,
                       tolMesh=1e-10, tolJoint=0.01)

    NvarInput = 1
    NvarOutput = 1
    # with open('finaloneshot_1000.pkl','rb') as file2:
    # files2=open('finaloneshot_1000.pkl','rb')
    #     model=pickle.load(file2).to(device)
    # model=enaself_model().to(device)
    # print(model)


    # model.load_state_dict(torch.load('model_init_best.pth'))
    # model = USCNN(h, nx, ny, NvarInput, NvarOutput, initWay=None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchSize = 2
    nEpochs = 2000
    lr = 0.001
    Ns = 1
    nu = 0.01
    criterion = nn.MSELoss()
    padSingleSide = 1
    udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
    ParaList = [1, 7]
    caseName = ['TemplateCase0', 'TemplateCase6']
    OFV_sb = []
    for name in caseName:
        OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, name + '/30/C',
                                                    [name + '/30/T'],
                                                    [0, 1, 0, 1], 0.0, False)
        OFX = OFPic[:, :, 0]
        OFY = OFPic[:, :, 1]
        OFV = OFPic[:, :, 2]
        OFV_sb_Temp = np.zeros(OFV.shape)
        for i in range(nx):
            for j in range(ny):
                dist = (myMesh.x[j, i] - OFX) ** 2 + (myMesh.y[j, i] - OFY) ** 2
                idx_min = np.where(dist == dist.min())
                OFV_sb_Temp[j, i] = OFV[idx_min]
        OFV_sb.append(OFV_sb_Temp)
    train_set = FixGeoDataset(ParaList, myMesh, OFV_sb)
    training_data_loader = DataLoader(dataset=train_set,
                                      batch_size=batchSize)
    # MRes = []
    # EV = []
    value = 10
    all_result=[]
    for i in range(1):
        model = USCNN(params, h, nx, ny, NvarInput, NvarOutput).to(device)
        MRes = []
        EV = []
        # model.load_state_dict(torch.load('model_init_best_0-2.pth'))
        # model = enaself_model().to(device)

        # model=enaself_model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(0, nEpochs + 1):
            startTime = time.time()
            xRes = 0
            yRes = 0
            mRes = 0
            eU = 0
            eV = 0
            eP = 0
            for iteration, batch in enumerate(training_data_loader):
                [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(batch)
                optimizer.zero_grad()
                output = model(Para)
                output_pad = udfpad(output)
                outputV = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                         output_pad.shape[2],
                                                         output_pad.shape[3])
                for j in range(batchSize):
                    # Impose BC
                    outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
                        outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
                    outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
                        outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
                    outputV[j, 0, :, -padSingleSide:] = 0
                    outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]
                dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
                d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
                dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
                d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
                continuity = (d2vdy2 + d2vdx2);
                loss = criterion(continuity, continuity * 0)
                loss.backward()
                optimizer.step()
                loss_mass = criterion(continuity, continuity * 0)
                mRes += loss_mass.item()
                eV = eV + torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))

                # if epoch % 1000 == 0 or epoch % nEpochs == 0:
                #     torch.save(model, str(epoch) + '.pth')
            print('Epoch is ', epoch)
            print("mRes Loss is", (mRes / len(training_data_loader)))
            print("eV Loss is", (eV / len(training_data_loader)))
            mres = mRes / len(training_data_loader)
            ev = eV / len(training_data_loader)
            MRes.append(mres)
            EV.append(ev)
            if epoch > 500:
                if ev <= value:
                    value = ev
                    numepoch = epoch
                    print(numepoch)
                    torch.save(model.state_dict(), 'case2_model2_hpo1.pth')
        # EV=EV.cpu().detach().numpy()
        ev_final = EV[numepoch]
        ev_final = ev_final.cpu().detach().numpy()
        ev_final = float(ev_final)
        print('train error',ev_final)
        # nni.report_intermediate_result(1 / ev_final)
        h = 0.01
        r = 0.5
        R = 1
        dtheta = 0
        OFBCCoord = Ofpp.parse_boundary_field('TemplateCase/30/C')
        # OFLOWC=OFBCCoord[b'low'][b'value']
        # OFUPC=OFBCCoord[b'up'][b'value']
        OFLEFTC = OFBCCoord[b'left'][b'value']
        OFRIGHTC = OFBCCoord[b'right'][b'value']

        leftX = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
        leftY = r * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
        rightX = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
        rightY = R * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))

        lowX = np.linspace(leftX[0], rightX[0], 49);
        lowY = lowX * 0 + np.sin(dtheta)
        upX = np.linspace(leftX[-1], rightX[-1], 49);
        upY = upX * 0 - np.sin(dtheta)

        ny = len(leftX);
        nx = len(lowX)

        myMesh = hcubeMesh(leftX, leftY, rightX, rightY,
                           lowX, lowY, upX, upY, h, False, True,
                           tolMesh=1e-10, tolJoint=0.01)
        criterion = nn.MSELoss()
        padSingleSide = 1
        udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
        ####
        ParaList = [2, 3, 4, 5, 6]
        caseName = ['TemplateCase1', 'TemplateCase2', 'TemplateCase3',
                    'TemplateCase4', 'TemplateCase5']
        OFV_sb = []
        for name in caseName:
            OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, name + '/30/C',
                                                        [name + '/30/T'],
                                                        [0, 1, 0, 1], 0.0, False)
            OFX = OFPic[:, :, 0]
            OFY = OFPic[:, :, 1]
            OFV = OFPic[:, :, 2]

            OFV_sb_Temp = np.zeros(OFV.shape)

            for i in range(nx):
                for j in range(ny):
                    dist = (myMesh.x[j, i] - OFX) ** 2 + (myMesh.y[j, i] - OFY) ** 2
                    idx_min = np.where(dist == dist.min())
                    OFV_sb_Temp[j, i] = OFV[idx_min]
            OFV_sb.append(OFV_sb_Temp)

        test_set = FixGeoDataset(ParaList, myMesh, OFV_sb)
        VelocityMagnitudeErrorRecord = []
        model.load_state_dict(torch.load('case2_model2_hpo1.pth',map_location=torch.device('cpu')))
        for i in range(len(ParaList)):
            [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(test_set[i])
            Para = Para.reshape(1, 1, Para.shape[0], Para.shape[1])
            truth = truth.reshape(1, 1, truth.shape[0], truth.shape[1])
            coord = coord.reshape(1, 2, coord.shape[2], coord.shape[3])
            print('i=', str(i))
            model.eval()
            # with torch.no_grad():
            output = model(Para)
            output_pad = udfpad(output)
            outputV = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            # Impose BC
            outputV[0, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
                outputV[0, 0, 1:2, padSingleSide:-padSingleSide])  # up outlet bc zero gradient
            outputV[0, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
                outputV[0, 0, -2:-1, padSingleSide:-padSingleSide])  # down inlet bc
            outputV[0, 0, :, -padSingleSide:] = 0  # right wall bc
            outputV[0, 0, :, 0:padSingleSide] = Para[0, 0, 0, 0]  # left  wall bc

            dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
            d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)

            dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
            d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
            # Calculate PDE Residual
            continuity = (d2vdy2 + d2vdx2);
            loss = criterion(continuity, continuity * 0)
            VelocityMagnitudeErrorRecord.append(torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0)))
        VErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
        print(VErrorNumpy)
        mean_error = float(np.mean(VErrorNumpy))
        print('test error:', mean_error)
        all_result.append(mean_error)
        nni.report_final_result(mean_error)
    print(all_result)
    print(np.mean(np.array(all_result)))






