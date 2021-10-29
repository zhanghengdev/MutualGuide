#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CEM(nn.Module):
    """Context Enhancement Module"""
    def __init__(self, channels, fea_channel, conv_block):
        super(CEM, self).__init__()
        self.conv1 = conv_block(channels[0], fea_channel, kernel_size=1, padding=0, relu=False)
        self.conv2 = nn.Sequential(
            conv_block(channels[1], fea_channel, kernel_size=1, padding=0, relu=False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_block(channels[1], fea_channel, kernel_size=1, padding=0, relu=False),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        C4_lat = self.conv1(inputs[0])
        C5_lat = self.conv2(inputs[1])
        Cglb_lat = self.conv3(inputs[1])
        return self.relu(C4_lat + C5_lat + Cglb_lat)


def fpn_feature_extractor(fpn_level, fea_channel, conv_block):
    layers = [conv_block(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1)]
    for _ in range(fpn_level - 1):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


def lateral_convs(fpn_level, fea_channel, conv_block):
    layers = []
    for _ in range(fpn_level):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=1))
    return nn.ModuleList(layers)


def fpn_convs(fpn_level, fea_channel, conv_block):
    layers = []
    for _ in range(fpn_level):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


class FPNNeck(nn.Module):

    def __init__(self, fpn_level, channels, fea_channel, conv_block):
        super(FPNNeck, self).__init__()
        self.fpn_level = fpn_level
        self.ft_module = CEM(channels, fea_channel, conv_block)
        self.pyramid_ext = fpn_feature_extractor(self.fpn_level, fea_channel, conv_block)
        self.lateral_convs = lateral_convs(self.fpn_level, fea_channel, conv_block)
        self.fpn_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)
        
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
        return fpn_fea

