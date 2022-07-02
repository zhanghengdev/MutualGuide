#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
from .fpn_neck import FPNNeck, fpn_convs
from .ssd_neck import fpn_extractor


class PAFPNNeck(FPNNeck):
    def __init__(
        self,
        fpn_level: int,
        channels: list,
        fea_channel: int,
        conv_block: nn.Module,
    ) -> None:
        FPNNeck.__init__(self, fpn_level, channels, fea_channel, conv_block)

        self.downsample_convs = fpn_extractor(self.fpn_level, fea_channel, conv_block)
        self.pafpn_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)

    def forward(
        self,
        x: list,
    ) -> list:
        fpn_fea = super().forward(x)
        for i in range(0, self.fpn_level - 1):
            fpn_fea[i + 1] = fpn_fea[i + 1] + self.downsample_convs[i](fpn_fea[i])
        fpn_fea = [pafpn_conv(x) for (x, pafpn_conv) in zip(fpn_fea, self.pafpn_convs)]
        return fpn_fea
