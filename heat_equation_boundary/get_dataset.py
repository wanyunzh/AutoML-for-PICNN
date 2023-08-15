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

class Mydataset(Dataset):
    def __init__(self,Para,myMesh,g_truth):
        self.Para=Para
        self.myMesh=myMesh
        self.g_truth=g_truth
    def __len__(self):
        return len(self.Para)
    def __getitem__(self, idx):
        mesh=self.myMesh
        x=mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        J = mesh.J_ho
        Jinv = mesh.Jinv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cordinate= np.concatenate((x[np.newaxis, :], y[np.newaxis, :]), axis=0)
        Temp_in=np.ones(x.shape[0])*self.Para[idx]
        Temp_out=np.zeros(x.shape[0])
        input=np.linspace(Temp_in,Temp_out,x.shape[1]).T
        return [input,cordinate,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta,self.g_truth[idx]]




def data_from_OF(nx,ny,t_case):
    file_c=t_case+'/30/C'
    file_t=t_case+'/30/T'
    truth_all=readScalarFromFile(file_t)
    truth_xyz=readVectorFromFile(file_c)
    truth_xyz[:,2]=truth_all
    o_x=truth_xyz[:,0].reshape(ny,nx,order='F')
    o_y = truth_xyz[:,1].reshape(ny, nx,order='F')
    o_truth=truth_xyz[:, 2].reshape(ny, nx,order='F')
    return o_x, o_y, o_truth

def get_dataset(mode='train'):
    h = 0.01
    r = 0.5
    R = 1
    dtheta = 0
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
    myMesh = Mesh(leftX, leftY, rightX, rightY,
                       lowX, lowY, upX, upY, h,
                       tolMesh=1e-10)
    if mode=='train':
        Para = [1, 7]
        case = ['TemplateCase0', 'TemplateCase6']
    elif mode=='test':
        Para = [2, 3, 4, 5, 6]
        case = ['TemplateCase1', 'TemplateCase2', 'TemplateCase3',
                    'TemplateCase4', 'TemplateCase5']
    g_truth=[]
    for item in case:
        o_x,o_y,o_truth=data_from_OF(nx,ny,item)
        mesh_true=np.zeros(o_truth.shape)
        for i in range(ny):#276
            for j in range(nx):#49
                idx_min=np.argmin((myMesh.x[i][j]-o_x)**2+(myMesh.y[i][j]-o_y)**2)
                mesh_true[i][j]=o_truth.flat[idx_min]
        g_truth.append(mesh_true)
    data_set=Mydataset(Para,myMesh,g_truth)
    return data_set


