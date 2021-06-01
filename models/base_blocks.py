#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def BasicConv(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, scale_factor=1):
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)]
    if bn:
        layers.append(nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True))
    if relu:
        layers.append(nn.ReLU(inplace=False))
    if scale_factor > 1:
        layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    return nn.Sequential(*layers)
