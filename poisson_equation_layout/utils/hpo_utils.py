import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

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
        diff = 10.0 * (diff - min) / (max - min)
        return torch.mean(torch.abs(diff * (inputs - targets)))

class Get_loss(torch.nn.Module):
    def __init__(
            self, params,device,nx=200, length=0.1, bcs=None
    ):
        super(Get_loss, self).__init__()
        self.length = length
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        self.weight_nine = torch.Tensor([[[[0.05, 0.2, 0.05], [0.2, 0, 0.2], [0.05, 0.2, 0.05]]]])
        # Padding
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        self.stride = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * self.stride ** 2 
        self.cof_nine = 0.05 * self.stride ** 2 
        self.constraint=params['constraint']
        self.filter_size=params['kernel']
        self.sobel_h_3x3 = torch.FloatTensor(
            np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]) / 8.0).unsqueeze(0).unsqueeze(0)

        self.sobel_v_3x3 = self.sobel_h_3x3.transpose(-1, -2)

        self.sobel_v_5x5 = torch.FloatTensor(
            np.array([[-5, -4, 0, 4, 5],
                      [-8, -10, 0, 10, 8],
                      [-10, -20, 0, 20, 10],
                      [-8, -10, 0, 10, 8],
                      [-5, -4, 0, 4, 5]]) / 240.).unsqueeze(0).unsqueeze(0)
        self.sobel_h_5x5 = self.sobel_v_5x5.transpose(-1, -2)

    def laplace_fraction(self, x):
        if self.filter_size==2:
            return F.conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)
        if self.filter_size == 4:
            return F.conv2d(x, self.weight_nine.to(device=x.device), bias=None, stride=1, padding=0)

    def grad_h(self, image):
        filter_size = self.filter_size
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.sobel_v_3x3.to(image.device)  # 使用 .detach()
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.sobel_v_5x5.to(image.device)  # 使用 .detach()
        image = F.pad(image, _quadruple(replicate_pad), mode='reflect')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) / self.stride
        return grad

    def grad_v(self, image):
        filter_size = self.filter_size
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.sobel_h_3x3.to(image.device)  # 使用 .detach()
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.sobel_h_5x5.to(image.device)  # 使用 .detach()
        image = F.pad(image, _quadruple(replicate_pad), mode='reflect')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) / self.stride
        return grad

    def forward(self, input, output):
        # Source item
        if self.filter_size == 2:
            f = self.cof * input
        if self.filter_size==4:
            f=self.cof_nine*input
        # The nodes which are not in boundary
        G = torch.ones_like(output).detach()
        idx_start = round(self.bcs[0][0] * self.nx / self.length)
        idx_end = round(self.bcs[1][0] * self.nx / self.length)
        G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
        output_b=output[..., idx_start:idx_end, :1]
        loss_b=torch.mean(torch.abs(output_b-torch.zeros_like(output_b)))
        if self.filter_size==2 or self.filter_size==4:
            if self.constraint == 1 or self.constraint == 2:
                x = F.pad(output * G, [1, 1, 1, 1], mode='reflect')# *G to ensure the Dir boundary(绝热部分温度相对为0度)，reflect means Neumann boundary padding and Dirichlet for 边缘点
            else:
                x = F.pad(output, [1, 1, 1, 1], mode='reflect')
            x = G * (self.laplace_fraction(x) + f) # the residue loss is calculated without the values in Dirichlet BCs.
            return x,loss_b
        if self.filter_size==3 or self.filter_size==5:
            if self.constraint == 1 or self.constraint == 2:
                output=output*G
            else:
                output=output
            grad_h = self.grad_h(output[:, [0]])
            grad_v = self.grad_v(output[:, [0]])
            grad_hh = self.grad_h(grad_h[:, [0]])
            grad_vv = self.grad_v(grad_v[:, [0]])
            input_pred = grad_vv + grad_hh
            return G*(input_pred + input),loss_b


# def get_gradient(continuity, device,nx,length):
#     replicate_pad = 1
#     kernel_h = torch.FloatTensor(
#             np.array([[-1, -2, -1],
#                       [0, 0, 0],
#                       [1, 2, 1]]) / 8.0).unsqueeze(0).unsqueeze(0).to(device)
#
#     image = F.pad(continuity, _quadruple(replicate_pad), mode='reflect')
#     stride = length / (nx - 1)
#     grad_h = F.conv2d(image, kernel_h, stride=1, padding=0, bias=None) / stride
#     kernel_v = kernel_h.transpose(-1, -2)
#     grad_v = F.conv2d(image, kernel_v, stride=1, padding=0,
#                     bias=None) / stride
#
#     return grad_h, grad_v




