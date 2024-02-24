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



def traintest_NS(model_cls):
    # tmp=model_cls
    # # print('model_cls: ',tmp)
    # tmp1=model_cls()
    # print('model_cls(): ', tmp1)

    params = {
        'UNARY_OPS': 1,
        'WEIGHT_INIT': 0,
        'WEIGHT_OPS': 3,
        'gradient': 0,
        'kernel': 2,
    }
    import nni
    print(params)
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device)
    nxOF = 50
    nyOF = 50
    nx = nxOF + 2
    ny = nyOF + 2
    scalarList = [-0.1, 0.0, 0.1]
    SolutionList = []
    MeshList = []
    for scalar in scalarList:
        OFcaseName_ = './TemplateCase' + str(scalar)
        nx = nxOF + 2
        ny = nyOF + 2
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
        Vmag_True = np.sqrt(OFU ** 2 + OFV ** 2)
        SolutionList.append(OFPic[:, :, 2:])
    batchSize = len(scalarList)

    nEpochs = 20001
    lr = 0.001
    Ns = 1
    nu = 0.01
    criterion = nn.MSELoss()
    padSingleSide = 1
    udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
    padSingleSide = 1
    udfpad = nn.ConstantPad2d([padSingleSide, padSingleSide, padSingleSide, padSingleSide], 0)
    train_set = VaryGeoDataset_PairedSolution(MeshList, SolutionList)
    training_data_loader = DataLoader(dataset=train_set,
                                      batch_size=batchSize)
    model = model_cls().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
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
    for epoch in range(0, nEpochs + 1):
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
                to4DTensor(batch, device)
            solutionTruth = torch.cat([Utrue, Vtrue, Ptrue], axis=1)
            optimizer.zero_grad()
            output = model(coord)
            output_pad = udfpad(output)
            if (epoch == 0) and (iteration) == 0:

                if params['WEIGHT_INIT'] == 1:
                    init_weight = torch.ones_like(output_pad)
                else:
                    init_weight = torch.zeros_like(output_pad)
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

            outputU = output_pad[:, 0, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            outputV = output_pad[:, 1, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])
            outputP = output_pad[:, 2, :, :].reshape(output_pad.shape[0], 1,
                                                     output_pad.shape[2],
                                                     output_pad.shape[3])

            for j in range(batchSize):
                outputU[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 0, -1, :].reshape(1,
                                                                                                           nx - 2 * padSingleSide)
                outputU[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0
                outputU[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
                outputU[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
                outputU[j, 0, 0, 0] = 0.5 * (outputU[j, 0, 0, 1] + outputU[j, 0, 1, 0])
                outputU[j, 0, 0, -1] = 0.5 * (outputU[j, 0, 0, -2] + outputU[j, 0, 1, -1])
                outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = output[j, 1, -1, :].reshape(1,
                                                                                                           nx - 2 * padSingleSide)
                outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = 0.4
                outputV[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 0
                outputV[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 0
                outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
                outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
                outputP[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
                outputP[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = output[j, 2, 0, :].reshape(1,
                                                                                                         nx - 2 * padSingleSide)
                outputP[j, 0, padSingleSide:-padSingleSide, -padSingleSide:] = output[j, 2, :, -1].reshape(
                    ny - 2 * padSingleSide, 1)
                outputP[j, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = output[j, 2, :, 0].reshape(
                    ny - 2 * padSingleSide, 1)
                outputP[j, 0, 0, 0] = 0.5 * (outputP[j, 0, 0, 1] + outputP[j, 0, 1, 0])
                outputP[j, 0, 0, -1] = 0.5 * (outputP[j, 0, 0, -2] + outputP[j, 0, 1, -1])
            kernel = params['kernel']
            diff_filter = Filter((outputV.shape[2], outputV.shape[3]), filter_size=kernel, device=device)

            dudx, dudy = pde_out(outputU, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            d2udx2, _ = pde_out(dudx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            _, d2udy2 = pde_out(dudy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            dvdx, dvdy = pde_out(outputV, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            d2vdx2, _ = pde_out(dvdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            _, d2vdy2 = pde_out(dvdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            dpdx, dpdy = pde_out(outputP, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            d2pdx2, _ = pde_out(dpdx, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            _, d2pdy2 = pde_out(dpdy, diff_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
            continuity, Xresidual, Yresidual = pde_residue(dudx, dudy, dvdx, dvdy, dpdx, dpdy, d2udx2, d2udy2,
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
            loss = loss_search + 5 * 1e-5 * (
                    gradient_loss_search1 + gradient_loss_search2 + gradient_loss_search3 + gradient_loss_search4 + gradient_loss_search5 + gradient_loss_search6)

            loss.backward()
            optimizer.step()
            Res = Res + loss.item()

            eU = eU + torch.sqrt(criterion(Utrue, output[:, 0:1, :, :]) / criterion(Utrue, Utrue * 0))
            eV = eV + torch.sqrt(criterion(Vtrue, output[:, 1:2, :, :]) / criterion(Vtrue, Vtrue * 0))
            eP = eP + torch.sqrt(criterion(Ptrue, output[:, 2:3, :, :]) / criterion(Ptrue, Ptrue * 0))
        print('Epoch is ', epoch)
        print("eU Loss is", (eU / len(training_data_loader)))
        print("eP Loss is", (eP / len(training_data_loader)))
        eu = eU / len(training_data_loader)
        ev = eV / len(training_data_loader)
        ep = eP / len(training_data_loader)

        EU.append(eu)
        EV.append(ev)
        EP.append(ep)

        if epoch >= 8000:
            if ep <= value:
                value = ep
                numepoch = epoch
                # print('min epoch: ',numepoch)
                torch.save(model.state_dict(), 'struct_search.pth')
    print('min epoch: ', numepoch)
    error_eu = EU[numepoch]
    error_ev = EV[numepoch]
    error_final = EP[numepoch]
    # ev_final = EV[2]
    error_final = error_final.cpu().detach().numpy()
    error_eu = error_eu.cpu().detach().numpy()
    error_ev = error_ev.cpu().detach().numpy()
    error_eu = float(error_eu)
    error_ev = float(error_ev)
    error_final = float(error_final)
    print('training l2 uerror', error_eu)
    print('training l2 verror', error_ev)
    print('training l2 perror', error_final)
    # nni.report_intermediate_result(float(error_final))
    # nni.report_final_result(float(error_final))
    nxOF = 50
    nyOF = 50

    scalarList1 = [-0.1,0.0,0.1]
    SolutionList = []
    MeshList = []
    for scalar in scalarList1:
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
    # model = torch.load('struct_search.pth')
    model.load_state_dict(torch.load('struct_search.pth'))

    for i in range(len(scalarList1)):
        [JJInv, coord, xi,
         eta, J, Jinv,
         dxdxi, dydxi,
         dxdeta, dydeta,
         Utrue, Vtrue, Ptrue] = \
            to4DTensor(test_set[i], device)
        solutionTruth = torch.cat([Utrue, Vtrue, Ptrue], axis=1)
        coord = coord.reshape(coord.shape[1], coord.shape[0], coord.shape[2], coord.shape[3])
        model.eval()
        output = model(coord)
        Vmag_True = torch.sqrt(Utrue ** 2 + Vtrue ** 2)
        Vmag_ = torch.sqrt(output[0, 0, :, :] ** 2 + output[0, 1, :, :] ** 2)

        VelocityMagnitudeErrorRecord.append(
            torch.sqrt(criterion(Vmag_True, Vmag_) / criterion(Vmag_True, Vmag_True * 0)))
        PErrorRecord.append(torch.sqrt(criterion(Ptrue, output[0, 2, :, :]) / criterion(Ptrue, Ptrue * 0)))
    VErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
    PErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in PErrorRecord])
    Verror_mean = np.mean(VErrorNumpy)
    Perror_mean = np.mean(PErrorNumpy)
    print('Intermediate Verror all:', VErrorNumpy)
    print('Intermeidate Perror all', PErrorNumpy)
    print('Intermeidate Verror:', Verror_mean)
    print('Intermeidate Perror', Perror_mean)
    out_final_report = float(1 / Perror_mean)
    nni.report_final_result(out_final_report)

    scalarList2 = [-0.075, -0.05, -0.025, 0.025, 0.05, 0.075]
    SolutionList = []
    MeshList = []
    for scalar in scalarList2:
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
    # model = torch.load('struct_search.pth')
    # model.load_state_dict(torch.load('hpo_case3.pth'))
    model.load_state_dict(torch.load('struct_search.pth'))

    for i in range(len(scalarList2)):
        [JJInv, coord, xi,
         eta, J, Jinv,
         dxdxi, dydxi,
         dxdeta, dydeta,
         Utrue, Vtrue, Ptrue] = \
            to4DTensor(test_set[i], device)
        solutionTruth = torch.cat([Utrue, Vtrue, Ptrue], axis=1)
        coord = coord.reshape(coord.shape[1], coord.shape[0], coord.shape[2], coord.shape[3])
        model.eval()
        output = model(coord)
        Vmag_True = torch.sqrt(Utrue ** 2 + Vtrue ** 2)
        Vmag_ = torch.sqrt(output[0, 0, :, :] ** 2 + output[0, 1, :, :] ** 2)

        VelocityMagnitudeErrorRecord.append(
            torch.sqrt(criterion(Vmag_True, Vmag_) / criterion(Vmag_True, Vmag_True * 0)))
        PErrorRecord.append(torch.sqrt(criterion(Ptrue, output[0, 2, :, :]) / criterion(Ptrue, Ptrue * 0)))
    VErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
    PErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in PErrorRecord])
    Verror_mean = np.mean(VErrorNumpy)
    Perror_mean = np.mean(PErrorNumpy)
    print('Fianl Verror all:', VErrorNumpy)
    print('Final Perror all', PErrorNumpy)
    print('Final Verror:', Verror_mean)
    print('Final Perror', Perror_mean)


