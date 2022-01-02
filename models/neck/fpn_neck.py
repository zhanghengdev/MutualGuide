#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssd_neck import SSDNeck


def lateral_convs(
    fpn_level: int,
    fea_channel: int,
    conv_block: nn.Module,
) -> nn.ModuleList:
    layers = []
    for _ in range(fpn_level):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=1))
    return nn.ModuleList(layers)


def fpn_convs(
    fpn_level: int,
    fea_channel: int,
    conv_block: nn.Module,
) -> nn.ModuleList:
    layers = []
    for _ in range(fpn_level):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


class FPNNeck(SSDNeck):
    def __init__(
        self,
        fpn_level: int,
        channels: list,
        fea_channel: int,
        conv_block: nn.Module,
    ) -> None:
        SSDNeck.__init__(self, fpn_level, channels, fea_channel, conv_block)
        
        self.lateral_convs = lateral_convs(self.fpn_level, fea_channel, conv_block)
        self.fpn_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)
        
    def forward(
        self,
        x: list,
    ) -> list:
        fpn_fea = super().forward(x)
        fpn_fea = [lateral_conv(x) for (x, lateral_conv) in zip(fpn_fea, self.lateral_convs)]
        for i in range(self.fpn_level - 1, 0, -1):
            fpn_fea[i - 1] = fpn_fea[i - 1] + F.interpolate(fpn_fea[i], scale_factor=2.0, mode='nearest')
        fpn_fea = [fpn_conv(x) for (x, fpn_conv) in zip(fpn_fea, self.fpn_convs)]
        return fpn_fea

