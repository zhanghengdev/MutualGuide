#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CEM(nn.Module):
    """Context Enhancement Module"""

    def __init__(
        self,
        channels: list,
        fea_channel: int,
        conv_block: nn.Module,
    ) -> None:
        super(CEM, self).__init__()

        for i, c in enumerate(channels):
            layer_name = f"conv{i+1}"
            if i == 0:
                layer = conv_block(c, fea_channel, kernel_size=1, act=False)
            else:
                layer = nn.Sequential(
                    conv_block(c, fea_channel, kernel_size=1, act=False),
                    nn.Upsample(scale_factor=2**i, mode="bilinear"),
                )
            self.add_module(layer_name, layer)

        layer_name = f"conv{i+2}"
        layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_block(channels[-1], fea_channel, kernel_size=1, act=False),
        )
        self.add_module(layer_name, layer)
        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        inputs: list,
    ) -> torch.Tensor:
        out = None
        for i, x in enumerate(inputs):
            layer = getattr(self, f"conv{i+1}")
            x = layer(x)
            out = x if out is None else x + out
        layer = getattr(self, f"conv{i+2}")
        context = layer(inputs[-1])
        return self.act(out + context)


def fpn_extractor(
    fpn_level: int,
    fea_channel: int,
    conv_block: nn.Module,
) -> nn.ModuleList:
    layers = []
    for _ in range(fpn_level - 1):
        layers.append(
            conv_block(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1)
        )
    return nn.ModuleList(layers)


class SSDNeck(nn.Module):
    def __init__(
        self,
        fpn_level: int,
        channels: list,
        fea_channel: int,
        conv_block: nn.Module,
    ) -> None:
        super(SSDNeck, self).__init__()
        self.fpn_level = fpn_level

        self.ft_module = CEM(channels, fea_channel, conv_block)
        self.pyramid_ext = fpn_extractor(self.fpn_level, fea_channel, conv_block)

    def forward(
        self,
        x: list,
    ) -> list:
        x = self.ft_module(x)
        fpn_fea = [x]
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)
        return fpn_fea
