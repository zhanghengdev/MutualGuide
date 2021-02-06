#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_blocks import BasicConv

def feature_transform_module(channels, fea_channel):
    layers = []
    for (i, channel) in enumerate(channels):
        layers.append(BasicConv(channel, fea_channel, kernel_size=1, padding=0, scale_factor=2 ** i))
    return nn.ModuleList(layers)


def fpn_feature_extractor(channels, fpn_level, fea_channel):
    layers = [BasicConv(fea_channel * len(channels), fea_channel, kernel_size=3, stride=1, padding=1)]
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

class FPNNeck(nn.Module):

    def __init__(self, fpn_level, channels, fea_channel):
        super(FPNNeck, self).__init__()
        self.fpn_level = fpn_level
        self.ft_module = feature_transform_module(channels, fea_channel)
        self.pyramid_ext = fpn_feature_extractor(channels, self.fpn_level, fea_channel)
        self.lateral_convs = lateral_convs(self.fpn_level, fea_channel)
        self.fpn_convs = fpn_convs(self.fpn_level, fea_channel)
        
    def forward(self, x):

        transformed_features = list()
        for (k, v) in zip(x, self.ft_module):
            transformed_features.append(v(k))
        x = torch.cat(transformed_features, 1)

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

