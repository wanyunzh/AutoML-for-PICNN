import numpy as np
from mesh import Mesh
import torch
import random
import torch.nn as nn
import Ofpp
import nni
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from foamFileOperation import readVectorFromFile,readScalarFromFile
from torch.utils.data import Dataset

def convert_to_4D_tensor(data_list,device):
    result_list = []
    for item in data_list:
        if len(item.shape) == 3:
            item = torch.tensor(item.reshape([item.shape[0], 1, item.shape[1], item.shape[2]]))
            result_list.append(item.float().to(device))
        else:
            item = torch.tensor(item)
            result_list.append(item.float().to(device))
    return result_list

def data_from_OF(nx,ny,file_name):
    file_c=file_name+'/1/C'
    file_f=file_name+'/1/f'
    truth_all=readScalarFromFile(file_f)
    truth_xyz=readVectorFromFile(file_c)
    truth_xyz[:,2]=truth_all
    o_x=truth_xyz[:,0].reshape(ny,nx,order='F')
    o_y = truth_xyz[:,1].reshape(ny, nx,order='F')
    o_truth=truth_xyz[:, 2].reshape(ny, nx,order='F')
    return o_x, o_y, o_truth

def get_dataset(device):
    h = 0.01
    OFBCCoord = Ofpp.parse_boundary_field('TemplateCase_4side/1/C')
    OFLOWC = OFBCCoord[b'low'][b'value']
    OFUPC = OFBCCoord[b'up'][b'value']
    OFLEFTC = OFBCCoord[b'left'][b'value']
    OFRIGHTC = OFBCCoord[b'right'][b'value']
    leftX = OFLEFTC[:, 0]
    leftY = OFLEFTC[:, 1]
    lowX = OFLOWC[:, 0]
    lowY = OFLOWC[:, 1]
    rightX = OFRIGHTC[:, 0]
    rightY = OFRIGHTC[:, 1]
    upX = OFUPC[:, 0]
    upY = OFUPC[:, 1]
    ny = len(leftX)
    nx = len(lowX)
    myMesh = Mesh(leftX, leftY, rightX, rightY,
                       lowX, lowY, upX, upY, h,
                       tolMesh=1e-10)


    file_name='TemplateCase_4side'
    o_x,o_y,o_truth=data_from_OF(nx,ny,file_name)
    g_truth = np.reshape(np.loadtxt('TemplateCase_4side/GT.txt').T, (1000,ny, nx),order='F')
    para_field = np.reshape(np.loadtxt('TemplateCase_4side/FI.txt').T, (1000,ny, nx),order='F')
    mesh_true=np.zeros(g_truth.shape)
    mesh_input=np.zeros(g_truth.shape)
    for item in range(mesh_true.shape[0]):
        for i in range(ny):#29
            for j in range(nx):#29
                idx_min=np.argmin((myMesh.x[i][j]-o_x)**2+(myMesh.y[i][j]-o_y)**2)
                mesh_input[item,i,j]=para_field[item].flat[idx_min]
                mesh_true[item, i, j] = g_truth[item].flat[idx_min]

    train_set = list(zip(mesh_input,mesh_true))
    [dydeta, dydxi, dxdxi, dxdeta, Jinv] = \
        convert_to_4D_tensor([myMesh.dydeta_ho, myMesh.dydxi_ho,
                    myMesh.dxdxi_ho, myMesh.dxdeta_ho,
                    myMesh.Jinv_ho],device)
    return train_set,dydeta, dydxi, dxdxi, dxdeta, Jinv



    # data_set=Mydataset(Para,myMesh,g_truth)
    # return data_set


