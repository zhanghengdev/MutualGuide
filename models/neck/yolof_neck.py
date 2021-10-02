#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.base_blocks import BasicConv

class BasicBlock(nn.Module):

    def __init__(self, channels, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
                    BasicConv(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                    BasicConv(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, relu=False),
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(
                x + self.conv(x)
            )

class YOLOFNeck(nn.Module):

    def __init__(self, channels, fea_channel, block_dilations=[2, 4, 6, 8]):
        super(YOLOFNeck, self).__init__()
        self.lateral_conv = BasicConv(channels[1], fea_channel, kernel_size=1, relu=False)
        self.context_conv = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    BasicConv(channels[1], fea_channel, kernel_size=1, relu=False),
                )
        self.relu = nn.ReLU(inplace=True)
        encoder_blocks = []
        for i in block_dilations:
            encoder_blocks.append(BasicBlock(fea_channel, dilation=i))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def forward(self, x):
        x=x[1]
        out = self.relu(
                self.lateral_conv(x) + self.context_conv(x)
            )
        out = self.dilated_encoder_blocks(out)
        return [out]
