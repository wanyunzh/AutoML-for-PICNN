import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

h=0.01
class P_OHEM(torch.nn.Module):
    """
    Weighted Loss
    """

    def __init__(self, loss_fun, weight=None):
        super(P_OHEM, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def forward(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        if inputs.ndim == 4:
            min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        elif inputs.ndim == 3:
            min, max = min.reshape(diff.shape[0], 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1).expand(diff.shape)
        diff =1.0 * (diff - min) / (max - min)
        continuity1=diff * (inputs - targets)
        criterion = nn.MSELoss()
        out=criterion(continuity1, continuity1 * 0)
        return out

def conv_gpinn(output, sobel_filter):

    if sobel_filter.filter_size==2 or sobel_filter.filter_size==4:
        grad_h = sobel_filter.grad_h_2(output[:, [0]])
        grad_v = sobel_filter.grad_v_2(output[:, [0]])
    if sobel_filter.filter_size==3 or sobel_filter.filter_size==5:
        grad_h = sobel_filter.grad_h(output[:, [0]])
        grad_v = sobel_filter.grad_v(output[:, [0]])

    return grad_h,grad_v




def pde_residue(input,output, sobel_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv):
    scalefactor = 1
    dvdx, dvdy = get_gradient(output, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
    d2vdx2, _ = get_gradient(dvdx, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
    _, d2vdy2 = get_gradient(dvdy, sobel_filter, dydeta, dydxi, dxdxi, dxdeta, Jinv)
    continuity = (d2vdy2 + d2vdx2) + input * scalefactor
    return continuity

def pde_out(output, sobel_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv):
    dvdx,dvdy = get_gradient(output, sobel_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv)
    return dvdx,dvdy


def get_gradient(f, sobel_filter,dydeta, dydxi, dxdxi,dxdeta,Jinv):

    order=sobel_filter.filter_size
    if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
        dfdxi = sobel_filter.grad_h_2(f[:, [0]], filter_size=order)
        dfdeta = sobel_filter.grad_v_2(f[:, [0]], filter_size=order)
        dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
        dfdxi = sobel_filter.grad_h_2(f[:, [0]], filter_size=order)
        dfdeta = sobel_filter.grad_v_2(f[:, [0]], filter_size=order)
        dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)

    if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
        dfdxi = sobel_filter.grad_h(f[:, [0]], filter_size=order)
        dfdeta = sobel_filter.grad_v(f[:, [0]], filter_size=order)
        dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
        dfdxi = sobel_filter.grad_h(f[:, [0]], filter_size=order)
        dfdeta = sobel_filter.grad_v(f[:, [0]], filter_size=order)
        dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdx,dfdy

class Filter():

    def __init__(self, imsize, correct=True, filter_size=2,device='cpu'):

        self.sobel_h_3x3 = torch.FloatTensor(
            np.array([[-1, -2, -1],
                     [ 0, 0, 0],
                     [1, 2, 1]]) / 8.0).unsqueeze(0).unsqueeze(0).to(device)

        self.sobel_v_3x3 = self.sobel_h_3x3.transpose(-1, -2)

        self.sobel_v_5x5 = torch.FloatTensor(
                    np.array([[-5, -4, 0, 4, 5],
                                [-8, -10, 0, 10, 8],
                                [-10, -20, 0, 20, 10],
                                [-8, -10, 0, 10, 8],
                                [-5, -4, 0, 4, 5]]) / 240.).unsqueeze(0).unsqueeze(0).to(device)
        self.sobel_h_5x5 = self.sobel_v_5x5.transpose(-1, -2)

        modifier_h = np.eye(imsize[1])
        modifier_h[0:2, 0] = np.array([4, -1])
        modifier_h[-2:, -1] = np.array([-1, 4])
        self.modifier_h = torch.FloatTensor(modifier_h).to(device)
        modifier_v = np.eye(imsize[0])
        modifier_v[0:2, 0] = np.array([4, -1])
        modifier_v[-2:, -1] = np.array([-1, 4])
        self.modifier_v = torch.FloatTensor(modifier_v).to(device)
        self.correct = correct
        self.filter_size = filter_size


    def grad_h(self, image, filter_size):

        if filter_size == 3:
            padding_num = 1
            kernel = self.sobel_v_3x3
        elif filter_size == 5:
            padding_num = 2
            kernel = self.sobel_v_5x5
        image = F.pad(image, _quadruple(padding_num), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) /h
        # modify the boundary based on forward & backward finite difference (three points)
        # forward [-3, 4, -1], backward [3, -4, 1]
        if self.correct:
            return torch.matmul(grad, self.modifier_h)
        else:
            return grad

    def grad_h_2(self,f,filter_size):
        if filter_size == 2:
            dfdxi_internal = (f[:, :, :, 2:] - f[:, :, :, 0:-2]) / 2 / h
            dfdxi_left = (-3 * f[:, :, :, 0:-2] + 4 * f[:, :, :, 1:-1] - 1 * f[:, :, :, 2:]) / 2 / h
            dfdxi_right = (3 * f[:, :, :, 2:] - 4 * f[:, :, :, 1:-1] + 1 * f[:, :, :, 0:-2]) / 2 / h
            dfdx = torch.cat((dfdxi_left[:, :, :, 0:1], dfdxi_internal, dfdxi_right[:, :, :, -1:]), 3)
        #below is gradient along horizontal axis(x axis)
        if filter_size == 4:
            dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 /h
            dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :,3:]) / 6 /h
            dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :,0:-3]) / 6 /h
            dfdx = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
        return dfdx

    def grad_v_2(self,f,filter_size):
        # below is gradient along vertical axis (y axis)
        if filter_size == 2:
            dfdeta_internal = (f[:, :, 2:, :] - f[:, :, 0:-2, :]) / 2 / h
            dfdeta_low = (-3 * f[:, :, 0:-2, :] + 4 * f[:, :, 1:-1, :] - 1 * f[:, :, 2:, :]) / 2 / h
            dfdeta_up = (3 * f[:, :, 2:, :] - 4 * f[:, :, 1:-1, :] + 1 * f[:, :, 0:-2, :]) / 2 / h
            dfdy = torch.cat((dfdeta_low[:, :, 0:1, :], dfdeta_internal, dfdeta_up[:, :, -1:, :]), 2)
        if filter_size == 4:
            dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4, :]) / 12 /h
            dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:,:]) / 6 /h
            dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3, :]) / 6 /h
            dfdy = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)
        return dfdy

    def grad_v(self, image, filter_size):
        image_height = image.shape[-2]
        if filter_size == 3:
            padding_num = 1
            kernel = self.sobel_h_3x3
        elif filter_size == 5:
            padding_num = 2
            kernel = self.sobel_h_5x5
        image = F.pad(image, _quadruple(padding_num), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0,
            bias=None) /h
        # modify the boundary based on forward & backward finite difference
        if self.correct:
            return torch.matmul(self.modifier_v.t(), grad)
        else:
            return grad
