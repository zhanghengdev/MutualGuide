#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.base_blocks import *


class CEM(nn.Module):

    """Context Enhancement Module"""

    def __init__(self, channels, fea_channel=256, num_classes=20):
        super(CEM, self).__init__()
        self.cv1 = BasicConv(channels[0], fea_channel, kernel_size=1,
                             padding=0)
        self.cv2 = BasicConv(channels[1], fea_channel, kernel_size=1,
                             padding=0, scale_factor=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cv3 = BasicConv(channels[1], fea_channel, kernel_size=1,
                             padding=0)
        self.fc = nn.Linear(fea_channel, num_classes)

    def forward(self, inputs):
        C4_lat = self.cv1(inputs[0])
        C5_lat = self.cv2(inputs[1])
        Cglb_lat = self.cv3(self.gap(inputs[1]))
        fc_lat = torch.flatten(Cglb_lat, 1)
        fc_lat = self.fc(fc_lat)
        return C4_lat + C5_lat + Cglb_lat, fc_lat


def fpn_feature_extractor(fpn_level, fea_channel=256):
    layers = [BasicConv(fea_channel, fea_channel, kernel_size=3,
              stride=1, padding=1)]
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


def lateral_convs(fpn_level, fea_channel=256):
    layers = []
    for _ in range(fpn_level):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=1))
    return nn.ModuleList(layers)


def fpn_convs(fpn_level, fea_channel=256):
    layers = []
    for _ in range(fpn_level):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


def downsample_convs(fpn_level, fea_channel=256):
    layers = []
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


def multibox(
    fpn_level,
    num_anchors,
    num_classes,
    fea_channel=256,
    ):
    (loc_layers, conf_layers) = ([], [])
    loc_channel = num_anchors * 4
    cls_channel = num_anchors * num_classes
    for _ in range(fpn_level):
        loc_layers += [nn.Conv2d(fea_channel, loc_channel,
                       kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cls_channel,
                        kernel_size=3, padding=1)]
    return (nn.ModuleList(loc_layers), nn.ModuleList(conf_layers))


class PAFPN(nn.Module):

    def __init__(
        self,
        size,
        num_classes,
        backbone,
        ):
        super(PAFPN, self).__init__()

        # Params

        self.fpn_level = (4 if size < 512 else 5)
        self.num_classes = num_classes - 1
        self.num_anchors = 6

        # Backbone network

        if backbone == 'vgg16':
            from models.vgg_backbone import VGGBackbone
            self.backbone = VGGBackbone(pretrained=True)
            channels = (512, 512)
        elif backbone == 'resnet18':
            from models.resnet_backbone import resnet18
            self.backbone = resnet18(pretrained=True)
            channels = (256, 512)
        else:
            raise NotImplementedError

        # Extra layers

        self.ft_module = CEM(channels, self.num_classes)
        self.pyramid_ext = fpn_feature_extractor(self.fpn_level)
        self.lateral_convs = lateral_convs(self.fpn_level)
        self.fpn_convs = fpn_convs(self.fpn_level)
        self.downsample_convs = downsample_convs(self.fpn_level)
        self.pafpn_convs = fpn_convs(self.fpn_level)

        # Detection Head

        (self.loc, self.conf) = multibox(self.fpn_level,
                self.num_anchors, self.num_classes)

        bias_value = 0
        for modules in self.loc:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, bias_value)
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for modules in self.conf:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, bias_value)

    def forward(self, x):
        loc = list()
        conf = list()

        # backbone

        source_features = self.backbone(x)
        x, fc = self.ft_module(source_features)

        # detection

        fpn_fea = list()
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)

        # fpn

        laterals = [lateral_conv(x) for (x, lateral_conv) in
                    zip(fpn_fea, self.lateral_convs)]
        for i in range(self.fpn_level - 1, 0, -1):
            size = laterals[i - 1].size()[-2:]
            laterals[i - 1] = laterals[i - 1] \
                + F.interpolate(laterals[i], size=size, mode='nearest')
        fpn_fea = [fpn_conv(x) for (x, fpn_conv) in zip(laterals,
                   self.fpn_convs)]
        for i in range(0, self.fpn_level - 1):
            fpn_fea[i + 1] += self.downsample_convs[i](fpn_fea[i])
        pafpn_fea = [pafpn_conv(x) for (x, pafpn_conv) in zip(fpn_fea,
                     self.pafpn_convs)]

        # apply multibox head to source layers

        for (i, (x, l, c)) in enumerate(zip(pafpn_fea, self.loc,
                self.conf)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), fc)

def build_net(size=320, num_classes=20, backbone='vgg16'):

    if not size % 64 == 0:
        raise NotImplementedError('Error: Sorry size {} is not supported!'.format(size))

    return PAFPN(size, num_classes, backbone)

