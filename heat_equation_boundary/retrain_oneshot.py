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
from search_newstruct import USCNN_5
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset,FixGeoDataset
from mesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda
# from modelself import USCNN, USCNNSep
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from hpo_utils import *
from nni.retiarii import fixed_arch
import Ofpp
import nni
import random
from hb_oneshot import enaself_model

def dfdx(f, dydeta, dydxi, Jinv):
    # Equation 13（a） in paper
    h = 0.01
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
    h = 0.01
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

def to4DTensor(myList):
    # device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,4'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MyList = []
    for item in myList:
        if len(item.shape) == 3:
            item = torch.tensor(item.reshape([item.shape[0], 1, item.shape[1],item.shape[2]]))
            MyList.append(item.float().to(device))
        else:
            item = torch.tensor(item)
            MyList.append(item.float().to(device))
    return MyList

if __name__ == "__main__":
    params = {
        'constraint': 1,
        'loss function': 1,
        'kernel': 2,
    }
    # device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
    params1 = {
        'channel1': 16,
        'channel2': 32,
        'channel3': 16
    }
    model=enaself_model()
    # model = model_cls().to(device)
    # with fixed_arch('case2_cnn.json'):
    #     model = USCNN_5(params1,h, nx, ny, NvarInput, NvarOutput, initWay=None)
    #     print('final model:', model)
    model = model.to(device)
    torch.save(model.state_dict(), 'model_init.pth')
    batchSize = 2
    nEpochs = 1500
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
    all_result = []
    MRes = []
    EV = []
    value = 10
    id_list = []
    step = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(training_data_loader),
                                                    epochs=nEpochs, div_factor=2,
                                                    pct_start=0.5)

    for epoch in range(0, nEpochs):
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
            output = output[:, 0, :, :].reshape(output.shape[0], 1,
                                                output.shape[2],
                                                output.shape[3])
            output_pad = output.clone()
            outputV = output_pad
            loss_boundary = 0
            for j in range(batchSize):
                bc1 = outputV[j, 0, :, -padSingleSide:] - 0
                bc2 = outputV[j, 0, :, 0:padSingleSide] - Para[j, 0, 0, 0]
                loss_boundary = loss_boundary + 1 * (bc1 ** 2).sum() + 1 * (bc2 ** 2).sum()
            loss_boundary = loss_boundary / (2 * batchSize * bc1.shape[0])

            for j in range(batchSize):
                # Impose BC
                outputV[j, 0, -padSingleSide:, padSingleSide:-padSingleSide] = (
                    outputV[j, 0, 1:2, padSingleSide:-padSingleSide])
                outputV[j, 0, :padSingleSide, padSingleSide:-padSingleSide] = (
                    outputV[j, 0, -2:-1, padSingleSide:-padSingleSide])
                outputV[j, 0, :, -padSingleSide:] = 0
                outputV[j, 0, :, 0:padSingleSide] = Para[j, 0, 0, 0]

            if params['constraint'] == 0:
                kernel = params['kernel']
                sobel_filter = SobelFilter((output.shape[2], outputV.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0) + loss_boundary
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
                    if 200<epoch<1000 and epoch % 20 == 0:
                        step += 1
                        for j in range(batchSize):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(abs(out))
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % batchSize
                                if remain == (batchSize * iteration + j) % batchSize:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res ** 2
                    # Code For RAR+residue
                    loss_pde = loss / (
                            batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                        3])
                    loss = loss_pde + loss_boundary
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                    if epoch >= 200:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss + loss_boundary
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 200:
                        loss_fun = OHEMF12d(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss + loss_boundary
                    loss.backward()

            # soft+hard constraint
            if params['constraint'] == 1:
                kernel = params['kernel']
                sobel_filter = SobelFilter((output.shape[2], outputV.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0) + loss_boundary
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
                    if 200<epoch<1000 and epoch % 20 == 0:
                        step += 1
                        for j in range(batchSize):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(abs(out))
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % batchSize
                                if remain == (batchSize * iteration + j) % batchSize:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res ** 2
                    # Code For RAR+residue
                    loss_pde = loss / (
                            batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                        3])
                    loss = loss_pde + loss_boundary
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001* loss_g1 + 0.001 * loss_g2
                    if epoch >= 200:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss + loss_boundary
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 200:
                        loss_fun = OHEMF12d(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss + loss_boundary
                    loss.backward()
            # hard constraint
            if params['constraint'] == 2:
                kernel = params['kernel']
                sobel_filter = SobelFilter((outputV.shape[2], outputV.shape[3]), filter_size=kernel, device=device)
                if params['loss function'] == 0:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = criterion(continuity, continuity * 0)
                    loss.backward()
                if params['loss function'] == 1:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss = (continuity ** 2).sum()
                    if 200<epoch<1000 and epoch % 20 == 0:
                        step += 1
                        for j in range(batchSize):
                            out = continuity[j, 0, :, :].reshape(continuity.shape[2], continuity.shape[3])
                            index = torch.argmax(abs(out))
                            max_res = out.max()
                            id_list.append(index)
                            for num, id in enumerate(id_list):
                                remain = (num) % batchSize
                                if remain == (batchSize * iteration + j) % batchSize:
                                    add_res = out.view(-1)[id]
                                    loss = loss + add_res ** 2
                    # Code For RAR+residue
                    loss_pde = loss / (
                            batchSize * step + batchSize * continuity.shape[2] * continuity.shape[
                        3])
                    loss = loss_pde
                    loss.backward()
                if params['loss function'] == 2:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    drdx, drdy = pde_out(continuity, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    loss_g1 = criterion(drdx, drdx * 0)
                    loss_g2 = criterion(drdy, drdy * 0)
                    loss_res = criterion(continuity, continuity * 0)
                    loss_all = loss_res + 0.001 * loss_g1 + 0.001 * loss_g2
                    if epoch >= 200:
                        loss = loss_all
                    else:
                        loss = loss_res
                    loss = loss
                    loss.backward()
                if params['loss function'] == 3:
                    continuity = pde_residue(outputV, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
                    if epoch >= 200:
                        loss_fun = OHEMF12d(loss_fun=F.l1_loss)
                        loss = loss_fun(continuity, continuity * 0)
                    else:
                        loss = criterion(continuity, continuity * 0)
                    loss = loss
                    loss.backward()

            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_lr()
            if iteration == 0:
                print("lr:", current_lr)
            loss_mass = loss
            mRes += loss_mass.item()
            eV = eV + torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0))

            # if epoch % 1000 == 0 or epoch % nEpochs == 0:
            #     torch.save(model, str(epoch) + '.pth')
        print('Epoch is ', epoch)
        print("mRes Loss is", (mRes / len(training_data_loader)))
        print("eV Loss is", (eV / len(training_data_loader)))
        mres = mRes / len(training_data_loader)
        ev = eV / len(training_data_loader)
        if epoch % 200 == 0:
            nni.report_intermediate_result(float(ev))
        MRes.append(float(mres))
        EV.append(float(ev))
        if epoch > 500:
            if ev <= value:
                value = ev
                numepoch = epoch
                # print(numepoch)
                torch.save(model.state_dict(), 'hb_retrain_oneshot.pth')
    # EV=EV.cpu().detach().numpy()
    ev_final = EV[numepoch]
    # ev_final = EV[2]
    # ev_final = ev_final.cpu().detach().numpy()
    # ev_final = float(ev_final)
    print('numepoch:', numepoch)
    print('train error', ev_final)


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
    model.load_state_dict(torch.load('hb_retrain_oneshot.pth', map_location=torch.device('cpu')))
    for i in range(len(ParaList)):
        [Para, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta, truth] = to4DTensor(test_set[i])
        Para = Para.reshape(1, 1, Para.shape[0], Para.shape[1])
        truth = truth.reshape(1, 1, truth.shape[0], truth.shape[1])
        coord = coord.reshape(1, 2, coord.shape[2], coord.shape[3])
        print('i=', str(i))
        model.eval()
        # with torch.no_grad():
        output = model(Para)
        output_pad = output
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

        VelocityMagnitudeErrorRecord.append(torch.sqrt(criterion(truth, outputV) / criterion(truth, truth * 0)))
    VErrorNumpy = np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
    print(VErrorNumpy)
    mean_error = float(np.mean(VErrorNumpy))
    print('test error:', mean_error)
    nni.report_final_result( 1 / ev_final)
    final_result = 1 / ev_final

    # import matplotlib.pyplot as plt
    # # 训练损失列表和预测误差列表
    # # training_loss = [0.5, 0.4, 0.3, 0.2, 0.1]
    # # prediction_error = [0.2, 0.15, 0.12, 0.1, 0.08]
    # # 绘制训练损失曲线
    # fig1=plt.figure(1)
    # plt.plot(MRes, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    # plt.yscale('log')
    #
    # # 保存训练损失曲线图
    # fig1.savefig('retrain_loss_curve2.png',dpi=250)
    # fig2 = plt.figure(2)
    #
    # # 绘制预测误差曲线
    # plt.plot(EV, label='Prediction Error')
    # plt.xlabel('Epoch')
    # plt.ylabel('Error')
    # plt.title('Prediction Error of Validation set')
    # plt.legend()
    # plt.yscale('log')
    # # 保存预测误差曲线图
    # fig2.savefig('retrain_error_curve2.png',dpi=250)
    # print('drawing done')









