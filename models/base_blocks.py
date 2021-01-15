#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class BasicConv(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=True,
        scale_factor=1,
        ):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            )
        self.bn = (nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01,
                   affine=True) if bn else None)
        self.relu = (nn.ReLU(inplace=False) if relu else None)
        self.upsample = (nn.Upsample(scale_factor=scale_factor,
                         mode='nearest') if scale_factor > 1 else None)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight.detach().cpu().numpy()
            running_mean = branch.bn.running_mean.cpu().numpy()
            running_var = branch.bn.running_var.cpu().numpy()
            gamma = branch.bn.weight.detach().cpu().numpy()
            beta = branch.bn.bias.detach().cpu().numpy()
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            kernel = np.zeros((self.in_channels, self.in_channels, 3, 3))
            for i in range(self.in_channels):
                kernel[i, i, 1, 1] = 1
            running_mean = branch.running_mean.cpu().numpy()
            running_var = branch.running_var.cpu().numpy()
            gamma = branch.weight.detach().cpu().numpy()
            beta = branch.bias.detach().cpu().numpy()
            eps = branch.eps
        std = np.sqrt(running_var + eps)
        t = gamma / std
        t = np.reshape(t, (-1, 1, 1, 1))
        t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        kernel = np.zeros((kernel1x1.shape[0], kernel1x1.shape[1], 3, 3))
        kernel[:, :, 1:2, 1:2] = kernel1x1
        return kernel

    def repvgg_convert(self):
        kernel3x3, bias3x3 = self._fuse_bn(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

