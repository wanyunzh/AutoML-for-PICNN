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

def flux_diff(input, output, sobel_filter):

    if sobel_filter.filter_size==2 or sobel_filter.filter_size==4:
        grad_h = sobel_filter.grad_h_2(output[:, [0]])
        grad_v = sobel_filter.grad_v_2(output[:, [0]])
    if sobel_filter.filter_size==3 or sobel_filter.filter_size==5:
        grad_h = sobel_filter.grad_h(output[:, [0]])
        grad_v = sobel_filter.grad_v(output[:, [0]])
    flux1 = - input * grad_h
    flux2 = - input * grad_v
    out=(output[:, [1]] - flux1) ** 2 + (output[:, [2]] - flux2) ** 2
    # out = (output[:, [1]] - flux1) + (output[:, [2]] - flux2)
    return out.mean()
    # "The PDE residue of the first term of equation,flux1,flux2 can be obtained either" \
    # "from the gradient of u(first output) or form the output of the model(second and third output)"
    # return ((output[:, [1]] - flux1) ** 2
    #     + (output[:, [2]] - flux2) ** 2).mean()


def source_diff(output, sobel_filter):

    if sobel_filter.filter_size==2 or sobel_filter.filter_size==4:
        flux1_g = sobel_filter.grad_h_2(output[:, [1]])
        flux2_g = sobel_filter.grad_v_2(output[:, [2]])
    if sobel_filter.filter_size==3 or sobel_filter.filter_size==5:
        flux1_g = sobel_filter.grad_h(output[:, [1]])
        flux2_g = sobel_filter.grad_v(output[:, [2]])

    out=(flux1_g + flux2_g) ** 2
    # out = (flux1_g + flux2_g)
    return out.mean()
    


def flux_diff_kdim(input, output, sobel_filter):

    if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
        grad_h = sobel_filter.grad_h_2(output[:, [0]])
        grad_v = sobel_filter.grad_v_2(output[:, [0]])
    if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
        grad_h = sobel_filter.grad_h(output[:, [0]])
        grad_v = sobel_filter.grad_v(output[:, [0]])
    flux1 = - input * grad_h
    flux2 = - input * grad_v
    out = (output[:, [1]] - flux1) ** 2 + (output[:, [2]] - flux2) ** 2
    # out = (output[:, [1]] - flux1) + (output[:, [2]] - flux2)
    return out
    # "The PDE residue of the first term of equation,flux1,flux2 can be obtained either" \
    # "from the gradient of u(first output) or form the output of the model(second and third output)"
    # return ((output[:, [1]] - flux1) ** 2
    #     + (output[:, [2]] - flux2) ** 2).mean()


def source_diff_kdim(output, sobel_filter):

    if sobel_filter.filter_size == 2 or sobel_filter.filter_size == 4:
        flux1_g = sobel_filter.grad_h_2(output[:, [1]])
        flux2_g = sobel_filter.grad_v_2(output[:, [2]])
    if sobel_filter.filter_size == 3 or sobel_filter.filter_size == 5:
        flux1_g = sobel_filter.grad_h(output[:, [1]])
        flux2_g = sobel_filter.grad_v(output[:, [2]])

    out = (flux1_g + flux2_g) ** 2
    # out = (flux1_g + flux2_g)
    return out
    # leave the top and bottom row free, since flux2_g is almost 0,
    # don't want to enforce flux1_g to be also zero.
    # if use_tb:
    #     return ((flux1_g + flux2_g) ** 2).mean() # The second term of PDE residue of equation
    # else:
    #     return ((flux1_g + flux2_g) ** 2)[:, :, 1:-1, :].mean()


class Filter():

    def __init__(self, imsize, correct=True,filter_size=2,device='cpu'):

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

        modifier = np.eye(imsize)
        modifier[0:2, 0] = np.array([4, -1])
        modifier[-2:, -1] = np.array([-1, 4])
        self.modifier = torch.FloatTensor(modifier).to(device)
        self.correct = correct
        self.filter_size=filter_size


    def grad_h(self, image):
        filter_size=self.filter_size
        image_width = image.shape[-1]

        if filter_size == 3:
            padding_num = 1
            kernel = self.sobel_v_3x3
        elif filter_size == 5:
            padding_num = 2
            kernel = self.sobel_v_5x5
        image = F.pad(image, _quadruple(padding_num), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) * image_width
        # modify the boundary based on forward & backward finite difference (three points)
        # forward [-3, 4, -1], backward [3, -4, 1]
        if self.correct:
            return torch.matmul(grad, self.modifier)
        else:
            return grad

    def grad_h_2(self,f):
        order=self.filter_size
        image_width = f.shape[-1]
        if order == 4:
            dfdxi_internal = (-f[:, :, :, 4:] + 8 * f[:, :, :, 3:-1] - 8 * f[:, :, :, 1:-3] + f[:, :, :, 0:-4]) / 12 *image_width
            dfdxi_left = (-11 * f[:, :, :, 0:-3] + 18 * f[:, :, :, 1:-2] - 9 * f[:, :, :, 2:-1] + 2 * f[:, :, :,3:]) / 6 *image_width
            dfdxi_right = (11 * f[:, :, :, 3:] - 18 * f[:, :, :, 2:-1] + 9 * f[:, :, :, 1:-2] - 2 * f[:, :, :,0:-3]) / 6 *image_width
            dfdx = torch.cat((dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3)
        if order == 2:
            dfdxi_internal = (f[:, :, :, 2:] - f[:, :, :, 0:-2]) / 2 *image_width
            dfdxi_left = (-3 * f[:, :, :, 0:-2] + 4 * f[:, :, :, 1:-1] - 1 * f[:, :, :, 2:]) / 2 *image_width
            dfdxi_right = (3 * f[:, :, :, 2:] - 4 * f[:, :, :, 1:-1] + 1 * f[:, :, :, 0:-2]) / 2 *image_width
            dfdx = torch.cat((dfdxi_left[:, :, :, 0:1], dfdxi_internal, dfdxi_right[:, :, :, -1:]), 3)
        return dfdx

    def grad_v_2(self,f):
        # below is gradient along vertical axis (y axis)
        image_width = f.shape[-2]
        order=self.filter_size
        if order == 4:
            dfdeta_internal = (-f[:, :, 4:, :] + 8 * f[:, :, 3:-1, :] - 8 * f[:, :, 1:-3, :] + f[:, :, 0:-4,
                                                                                               :]) / 12 *image_width
            dfdeta_low = (-11 * f[:, :, 0:-3, :] + 18 * f[:, :, 1:-2, :] - 9 * f[:, :, 2:-1, :] + 2 * f[:, :, 3:,
                                                                                                      :]) / 6 *image_width
            dfdeta_up = (11 * f[:, :, 3:, :] - 18 * f[:, :, 2:-1, :] + 9 * f[:, :, 1:-2, :] - 2 * f[:, :, 0:-3,
                                                                                                  :]) / 6 *image_width
            dfdy = torch.cat((dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2)

        if order == 2:
            dfdeta_internal = (f[:, :, 2:, :] - f[:, :, 0:-2, :]) / 2 *image_width
            dfdeta_low = (-3 * f[:, :, 0:-2, :] + 4 * f[:, :, 1:-1, :] - 1 * f[:, :, 2:, :]) / 2 *image_width
            dfdeta_up = (3 * f[:, :, 2:, :] - 4 * f[:, :, 1:-1, :] + 1 * f[:, :, 0:-2, :]) / 2 *image_width
            dfdy = torch.cat((dfdeta_low[:, :, 0:1, :], dfdeta_internal, dfdeta_up[:, :, -1:, :]), 2)
        return dfdy

    def grad_v(self, image):
        filter_size=self.filter_size
        image_height = image.shape[-2]
        if filter_size == 3:
            padding_num = 1
            kernel = self.sobel_h_3x3
        elif filter_size == 5:
            padding_num = 2
            kernel = self.sobel_h_5x5
        image = F.pad(image, _quadruple(padding_num), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0,
            bias=None) * image_height
        # modify the boundary based on forward & backward finite difference
        if self.correct:
            return torch.matmul(self.modifier.t(), grad)
        else:
            return grad


def loss_origin(input, output, sobel_filter):
    loss_pde = flux_diff(input, output, sobel_filter) \
               + source_diff(output, sobel_filter)
    return loss_pde

def loss_gpinn(input, output, sobel_filter,epoch):
    out1 = flux_diff_kdim(input, output, sobel_filter)
    out2 = source_diff_kdim(output, sobel_filter)
    if epoch >= 10:
        loss_pde0 = flux_diff(input, output, sobel_filter) \
                    + source_diff(output, sobel_filter)
        dr1dx, dr1dy = conv_gpinn(out1, sobel_filter)
        dr2dx, dr2dy = conv_gpinn(out2, sobel_filter)
        criterion = nn.MSELoss()
        dr1dx, dr1dy = criterion(dr1dx, torch.zeros_like(dr1dx)), criterion(dr1dy,
                                                                            torch.zeros_like(dr1dy))
        dr2dx, dr2dy = criterion(dr2dx, torch.zeros_like(dr2dx)), criterion(dr2dy,
                                                                            torch.zeros_like(dr2dy))

        loss_pde = loss_pde0 + 0.05 * (dr1dx + dr1dy + dr2dx + dr2dy)
    else:
        loss_pde = flux_diff(input, output, sobel_filter) \
                   + source_diff(output, sobel_filter)
    return loss_pde


def loss_ohem(input, output, sobel_filter,epoch):
    out1 = flux_diff_kdim(input, output, sobel_filter)
    out2 = source_diff_kdim(output, sobel_filter)
    if epoch >= 10:
        loss_fun = P_OHEM(loss_fun=F.l1_loss)
        loss1 = loss_fun(out1, out1 * 0)
        loss2 = loss_fun(out2, out2 * 0)
        loss_pde = loss1 + loss2
    else:
        loss_pde = flux_diff(input, output, sobel_filter) \
                   + source_diff(output, sobel_filter)
    return loss_pde

