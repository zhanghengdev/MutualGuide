#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def switch_to_deploy(self):
        if self.bn is None:
            return

        fusedconv = (
            nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(self.conv.weight.device)
        )

        # prepare filters
        w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
        w_bn = torch.diag(self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = (
            torch.zeros(self.conv.weight.size(0), device=self.conv.weight.device)
            if self.conv.bias is None
            else self.conv.bias
        )
        b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
            torch.sqrt(self.bn.running_var + self.bn.eps)
        )
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        self.conv = fusedconv
        self.bn = None


class DepthwiseConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(DepthwiseConv, self).__init__()

        if kernel_size == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation, groups=in_planes, bias=bias),
                    nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True),
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                )
            
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

