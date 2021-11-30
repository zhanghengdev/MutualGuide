#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn_neck import FPNNeck


def fpn_convs(fpn_level, fea_channel, conv_block):
    layers = []
    for _ in range(fpn_level):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


def downsample_convs(fpn_level, fea_channel, conv_block):
    layers = []
    for _ in range(fpn_level - 1):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


class PAFPNNeck(FPNNeck):

    def __init__(self, fpn_level, channels, fea_channel, conv_block):
        FPNNeck.__init__(self, fpn_level, channels, fea_channel, conv_block)
        self.downsample_convs = downsample_convs(self.fpn_level, fea_channel, conv_block)
        self.pafpn_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)


    def forward(self, x):
        fpn_fea = super().forward(x)
        for i in range(0, self.fpn_level - 1):
            fpn_fea[i + 1] = fpn_fea[i + 1] + self.downsample_convs[i](fpn_fea[i])
        fpn_fea = [pafpn_conv(x) for (x, pafpn_conv) in zip(fpn_fea, self.pafpn_convs)]
        return fpn_fea

