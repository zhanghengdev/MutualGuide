#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from .neck import *
from .backbone import *
from .base_blocks import *


def multibox(
    fpn_level: int,
    num_anchors: int,
    num_classes: int,
    fea_channel: int,
    dis_channel: int,
    conv_block: nn.Module,
) -> tuple:
    loc_layers, conf_layers, dist_layers = list(), list(), list()
    for _ in range(fpn_level):
        loc_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, num_anchors * 4, 1),
            )
        )
        conf_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, num_anchors * num_classes, 1),
            )
        )
        dist_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, dis_channel, 1),
            )
        )
    return (
        nn.ModuleList(loc_layers),
        nn.ModuleList(conf_layers),
        nn.ModuleList(dist_layers),
    )


class Detector(nn.Module):
    """Student Detector Model"""

    def __init__(
        self,
        base_size: int,
        num_classes: int,
        backbone: str,
        neck: str,
        mode: str,
    ) -> None:
        super(Detector, self).__init__()

        # Params
        self.num_classes = num_classes
        self.num_anchors = 6
        self.mode = mode
        self.fpn_level = 3 if base_size <= 640 else 4

        # Backbone network
        if "vgg" in backbone:
            self.backbone = VGGBackbone(version=backbone)
            self.conv_block = BasicConv
        elif "resnet" in backbone:
            self.backbone = ResNetBackbone(version=backbone)
            self.conv_block = BasicConv
        elif "repvgg" in backbone:
            self.backbone = REPVGGBackbone(version=backbone)
            self.conv_block = BasicConv
        elif "cspdarknet" in backbone:
            self.backbone = CSPDarkNetBackbone(version=backbone)
            self.conv_block = BasicConv
        elif "shufflenet" in backbone:
            self.backbone = ShuffleNetBackbone(version=backbone)
            self.conv_block = DepthwiseConv
        elif "efficientnet" in backbone:
            self.backbone = EfficientNetBackbone.from_name(backbone)
            self.conv_block = DepthwiseConv
        else:
            raise ValueError(
                "Error: Sorry backbone {} is not supported!".format(backbone)
            )

        # Neck width
        if self.conv_block is BasicConv:
            self.fea_channel = self.dis_channel = 256
        elif self.conv_block is DepthwiseConv:
            self.fea_channel = self.dis_channel = 128
        else:
            raise ValueError(
                "Error: Sorry conv_block {} is not supported!".format(self.conv_block)
            )

        # Neck network
        if neck == "ssd":
            neck_func = SSDNeck
        elif neck == "fpn":
            neck_func = FPNNeck
        elif neck == "pafpn":
            neck_func = PAFPNNeck
        else:
            raise ValueError("Error: Sorry neck {} is not supported!".format(neck))
        self.neck = neck_func(
            self.fpn_level,
            self.backbone.out_channels,
            self.fea_channel,
            self.conv_block,
        )

        # Detection Head
        (self.loc, self.conf, self.dist) = multibox(
            self.fpn_level,
            self.num_anchors,
            self.num_classes,
            self.fea_channel,
            self.dis_channel,
            self.conv_block,
        )

        bias_value = 0
        for modules in self.loc:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for modules in self.conf:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)

    def deploy(
        self,
    ) -> None:
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.eval()

    def _forward_func_tea(
        self,
        fp: list,
    ) -> dict:
        fea = list()
        loc = list()
        conf = list()
        for (x, l, c) in zip(fp, self.loc, self.conf):
            fea.append(x.permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {
            "loc": loc.view(loc.size(0), -1, 4),
            "conf": conf.view(conf.size(0), -1, self.num_classes),
            "feature": fea.view(conf.size(0), -1, self.dis_channel),
        }

    def _forward_func_stu(
        self,
        fp: list,
    ) -> dict:
        fea = list()
        loc = list()
        conf = list()
        for (x, l, c, d) in zip(fp, self.loc, self.conf, self.dist):
            fea.append(d(x).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {
            "loc": loc.view(loc.size(0), -1, 4),
            "conf": conf.view(conf.size(0), -1, self.num_classes),
            "feature": fea.view(conf.size(0), -1, self.dis_channel),
        }

    def _forward_func_nor(
        self,
        fp: list,
    ) -> dict:
        loc = list()
        conf = list()
        for (x, l, c) in zip(fp, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {
            "loc": loc.view(loc.size(0), -1, 4),
            "conf": conf.view(conf.size(0), -1, self.num_classes),
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        x = self.backbone(x)
        fp = self.neck(x)
        if self.mode == "teacher":
            return self._forward_func_tea(fp)
        elif self.mode == "student":
            return self._forward_func_stu(fp)
        elif self.mode == "normal":
            return self._forward_func_nor(fp)
        raise NotImplementedError
