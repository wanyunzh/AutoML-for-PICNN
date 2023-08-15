# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """
    def __init__(self, pool_type, kernel_size, stride, padding):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

    def forward(self, x):
        out = self.pool(x)
        return out

class Pool2d_conv(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """
    def __init__(self, channels_p, c_cur,pool_type, kernel_size, stride):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride)
        else:
            raise ValueError()
        layers = [
            nn.Conv2d(channels_p, c_cur, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(32, c_cur),
            nn.GELU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        out = self.pool(x)
        out=self.encode(out)
        return out

class Pool2d(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """
    def __init__(self, pool_type, kernel_size, stride):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride)
        else:
            raise ValueError()


    def forward(self, x):
        out = self.pool(x)
        return out


class StdConv(nn.Module):
    """
    Standard conv:  Conv-GeLU-GN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False,padding_mode='reflect'),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class upStdConv(nn.Module):
    """
    Standard conv:  Conv-GeLU-GN
    """
    def __init__(self, C_in, C_out, kernel_size, stride,padding):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, kernel_size,
                               stride,padding,output_padding=1),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

class DilConv(nn.Module):
    """
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False,padding_mode='reflect'),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class upDilConv(nn.Module):
    """
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """
    def __init__(self, C_in, C_out, kernel_size, stride,padding,dilation):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel_size, stride,padding,dilation=dilation, groups=C_in,
                      bias=False,output_padding=1),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in,padding_mode='reflect'),
            nn.Conv2d(C_in, C_out, kernel_size=1),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)


class upSepConv(nn.Module):
    """
    Depthwise separable conv.
    """

    def __init__(self, C_in, C_out, kernel_size, stride,padding):
        super().__init__()
        group = 32 if C_out % 32 == 0 else 16
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel_size, stride,padding,groups=C_in,output_padding=1),
            nn.Conv2d(C_in, C_out, kernel_size=1),
            nn.GroupNorm(group, C_out),
            nn.GELU()
        )


    def forward(self, x):
        return self.net(x)


class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class UpsamplingBilinear2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bilinear', align_corners=True)
class DropPath(nn.Module):
    def __init__(self, p=0.):
        """
        Drop path with probability.

        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1, 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask

        return x