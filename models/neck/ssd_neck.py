#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_blocks import BasicConv


class CEM(nn.Module):

    """Context Enhancement Module"""

    def __init__(self, channels, fea_channel):
        super(CEM, self).__init__()
        self.conv1 = BasicConv(channels[0], fea_channel, kernel_size=1, padding=0, relu=False)
        self.conv2 = nn.Sequential(
            BasicConv(channels[1], fea_channel, kernel_size=1, padding=0, relu=False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        C4_lat = self.conv1(inputs[0])
        C5_lat = self.conv2(inputs[1])
        return self.relu(C4_lat + C5_lat)


def fpn_feature_extractor(fpn_level, fea_channel):
    layers = [BasicConv(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1)]
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


class SSDNeck(nn.Module):

    def __init__(self, fpn_level, channels, fea_channel):
        super(SSDNeck, self).__init__()
        self.fpn_level = fpn_level
        self.ft_module = CEM(channels, fea_channel)
        self.pyramid_ext = fpn_feature_extractor(self.fpn_level, fea_channel)
        
    def forward(self, x):
        x = self.ft_module(x)
        fpn_fea = list()
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)
        return fpn_fea