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


class SSDNeck(nn.Module):

    def __init__(self, fpn_level, channels, fea_channel):
        super(SSDNeck, self).__init__()
        self.ft_module = feature_transform_module(channels, fea_channel)
        self.pyramid_ext = fpn_feature_extractor(channels, fpn_level, fea_channel)

    def forward(self, x):
        transformed_features = list()
        for (k, v) in zip(x, self.ft_module):
            transformed_features.append(v(k))
        x = torch.cat(transformed_features, 1)

        fpn_fea = list()
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)
        return fpn_fea

