#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    """Basic Convolution Module"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: bool = True,
        bn: bool = True,
    ) -> None:
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
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.act = nn.SiLU(inplace=True) if act else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def switch_to_deploy(
        self,
    ) -> None:
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
        w_bn = torch.diag(
            self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var))
        )
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
    """Depthwise Convolution Module"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        act: bool = True,
        bn: bool = True,
    ) -> None:
        super(DepthwiseConv, self).__init__()

        if kernel_size == 1:
            self.dconv = None
        else:
            kernel_size, padding = 5, 2
            self.dconv = BasicConv(
                in_planes,
                in_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_planes,
                bias=bias,
                act=False,
                bn=True,
            )
        self.pconv = BasicConv(
            in_planes,
            out_planes,
            kernel_size=1,
            act=act,
            bn=bn,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.dconv is not None:
            x = self.dconv(x)
        x = self.pconv(x)
        return x
