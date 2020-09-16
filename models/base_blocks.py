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

