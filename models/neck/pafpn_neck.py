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
        self.cv1 = BasicConv(channels[0], fea_channel, kernel_size=1, padding=0)
        self.cv2 = BasicConv(channels[1], fea_channel, kernel_size=1, padding=0, scale_factor=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cv3 = BasicConv(channels[1], fea_channel, kernel_size=1, padding=0)

    def forward(self, inputs):
        C4_lat = self.cv1(inputs[0])
        C5_lat = self.cv2(inputs[1])
        Cglb_lat = self.cv3(self.gap(inputs[1]))
        return C4_lat + C5_lat + Cglb_lat


def fpn_feature_extractor(fpn_level, fea_channel):
    layers = [BasicConv(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1)]
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


def lateral_convs(fpn_level, fea_channel):
    layers = []
    for _ in range(fpn_level):
        layers.append(BasicConv(fea_channel, fea_channel, kernel_size=1))
    return nn.ModuleList(layers)


def fpn_convs(fpn_level, fea_channel):
    layers = []
    for _ in range(fpn_level):
        layers.append(BasicConv(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


def downsample_convs(fpn_level, fea_channel):
    layers = []
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


class PAFPNNeck(nn.Module):

    def __init__(self, fpn_level, channels, fea_channel):
        super(PAFPNNeck, self).__init__()
        self.fpn_level = fpn_level
        self.ft_module = CEM(channels, fea_channel)
        self.pyramid_ext = fpn_feature_extractor(self.fpn_level, fea_channel)
        self.lateral_convs = lateral_convs(self.fpn_level, fea_channel)
        self.fpn_convs = fpn_convs(self.fpn_level, fea_channel)
        self.downsample_convs = downsample_convs(self.fpn_level, fea_channel)
        self.pafpn_convs = fpn_convs(self.fpn_level, fea_channel)


    def forward(self, x):
        x = self.ft_module(x)
        fpn_fea = list()
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)
        laterals = [lateral_conv(x) for (x, lateral_conv) in zip(fpn_fea, self.lateral_convs)]
        for i in range(self.fpn_level - 1, 0, -1):
            size = laterals[i - 1].size()[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=size, mode='nearest')
        fpn_fea = [fpn_conv(x) for (x, fpn_conv) in zip(laterals, self.fpn_convs)]
        for i in range(0, self.fpn_level - 1):
            fpn_fea[i + 1] += self.downsample_convs[i](fpn_fea[i])
        pafpn_fea = [pafpn_conv(x) for (x, pafpn_conv) in zip(fpn_fea, self.pafpn_convs)]
        return pafpn_fea

