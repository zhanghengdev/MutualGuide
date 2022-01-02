#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.base_blocks import BasicConv, DepthwiseConv


def multibox(
    fpn_level: int,
    num_anchors: int,
    num_classes: int,
    fea_channel: int,
    dis_channel: int,
    conv_block: nn.Module,
) -> tuple:
    loc_layers, conf_layers, dist_layers = list(), list(), list()
    loc_channel = num_anchors * 4
    cls_channel = num_anchors * num_classes
    for _ in range(fpn_level):
        loc_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, loc_channel, 1),
            )
        )
        conf_layers.append(
            nn.Sequential(
                conv_block(fea_channel, fea_channel, 3, padding=1),
                conv_block(fea_channel, fea_channel, 3, padding=1),
                nn.Conv2d(fea_channel, cls_channel, 1),
            )
        )
        dist_layers.append(
            nn.Conv2d(fea_channel, dis_channel, 1)
        )
    return (
        nn.ModuleList(loc_layers),
        nn.ModuleList(conf_layers),
        nn.ModuleList(dist_layers),
    )


class Detector(nn.Module):
    """ Student Detector Model """

    def __init__(
        self,
        base_size: int,
        num_classes: int,
        backbone: str,
        neck: str,
    ) -> None:
        super(Detector, self).__init__()

        # Params
        self.num_classes = num_classes - 1
        self.num_anchors = 6
        self.fpn_level = 4 if base_size < 512 else 5

        # Backbone network
        if backbone == 'resnet18':
            from models.backbone.resnet_backbone import ResNetBackbone
            self.backbone = ResNetBackbone(depth=18)
            self.backbone_channels = (256, 512)
            self.fea_channel = 256
            self.dis_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'vgg11':
            from models.backbone.vgg_backbone import VGGBackbone
            self.backbone = VGGBackbone(depth=11)
            self.backbone_channels = (512, 512)
            self.fea_channel = 256
            self.dis_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'repvgg-A0':
            from models.backbone.repvgg_backbone import REPVGGBackbone
            self.backbone = REPVGGBackbone(version='A0')
            self.backbone_channels = (192, 1280)
            self.fea_channel = 128
            self.dis_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'repvgg-A1':
            from models.backbone.repvgg_backbone import REPVGGBackbone
            self.backbone = REPVGGBackbone(version='A1')
            self.backbone_channels = (256, 1280)
            self.fea_channel = 256
            self.dis_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'repvgg-B1':
            from models.backbone.repvgg_backbone import REPVGGBackbone
            self.backbone = REPVGGBackbone(version='B1')
            self.backbone_channels = (512, 2048)
            self.fea_channel = 256
            self.dis_channel = 256
            self.conv_block = BasicConv
        elif backbone == 'shufflenet-0.5':
            from models.backbone.shufflenet_backbone import ShuffleNetBackbone
            self.backbone = ShuffleNetBackbone(width=0.5)
            self.backbone_channels = (96, 192)
            self.fea_channel = 128
            self.dis_channel = 128
            self.conv_block = DepthwiseConv
        else:
            raise ValueError('Error: Sorry backbone {} is not supported!'.format(backbone))

        # Neck network
        if neck == 'ssd':
            from models.neck.ssd_neck import SSDNeck
            self.neck = SSDNeck(self.fpn_level, self.backbone_channels, self.fea_channel, self.conv_block)
        elif neck == 'fpn':
            from models.neck.fpn_neck import FPNNeck
            self.neck = FPNNeck(self.fpn_level, self.backbone_channels, self.fea_channel, self.conv_block)
        elif neck == 'pafpn':
            from models.neck.pafpn_neck import PAFPNNeck
            self.neck = PAFPNNeck(self.fpn_level, self.backbone_channels, self.fea_channel, self.conv_block)
        else:
            raise ValueError('Error: Sorry neck {} is not supported!'.format(neck))

        # Detection Head
        (self.loc, self.conf, self.dist) = multibox(
            self.fpn_level, self.num_anchors, self.num_classes, self.fea_channel, self.dis_channel, self.conv_block,
        )
        bias_value = 0
        for modules in self.loc:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for modules in self.conf:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)

    def deploy(
        self,
    ) -> None:
        for module in self.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.eval()

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        x = self.backbone(x)
        fp = self.neck(x)
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
            'loc': loc.view(loc.size(0), -1, 4), 
            'conf': conf.view(conf.size(0), -1, self.num_classes),
            'feature': fea.view(conf.size(0), -1, self.dis_channel),
        }


    def forward_test(
        self,
        x: torch.Tensor,
    ) -> dict:
        x = self.backbone(x)
        fp = self.neck(x)
        loc = list()
        conf = list()
        for (x, l, c) in zip(fp, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {
            'loc': loc.view(loc.size(0), -1, 4), 
            'conf': conf.view(conf.size(0), -1, self.num_classes),
        }
