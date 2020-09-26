#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.base_blocks import *


def feature_transform_module(channels):
    layers = []
    for (i, channel) in enumerate(channels):
        layers.append(BasicConv(channel, 256, kernel_size=1, padding=0,
                      scale_factor=2 ** i))
    return nn.ModuleList(layers)


def pyramid_feature_extractor(fpn_level, channels):
    fea_channel = 256
    layers = [BasicConv(256 * len(channels), fea_channel,
              kernel_size=3, stride=1, padding=1)]
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


def lateral_convs(fpn_level):
    layers = []
    fea_channel = 256
    for _ in range(fpn_level):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=1))
    return nn.ModuleList(layers)


def fpn_convs(fpn_level):
    layers = []
    fea_channel = 256
    for _ in range(fpn_level):
        layers.append(BasicConv(fea_channel, fea_channel,
                      kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


def multibox(fpn_level, num_anchors, num_classes):
    (loc_layers, conf_layers) = ([], [])
    fea_channel = 256
    loc_channel = num_anchors * 4
    cls_channel = num_anchors * num_classes
    for _ in range(fpn_level):
        loc_layers += [nn.Conv2d(fea_channel, loc_channel,
                       kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cls_channel,
                        kernel_size=3, padding=1)]
    return (nn.ModuleList(loc_layers), nn.ModuleList(conf_layers))


class RetinaNet(nn.Module):

    def __init__(
        self,
        size,
        num_classes,
        backbone,
        ):
        super(RetinaNet, self).__init__()

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
            channels = (128, 256, 512)
        elif backbone == 'resnet50':
            from models.resnet_backbone import resnet50
            self.backbone = resnet50(pretrained=True)
            channels = (512, 1024, 2048)
        elif backbone == 'resnet101':
            from models.resnet_backbone import resnet101
            self.backbone = resnet101(pretrained=True)
            channels = (512, 1024, 2048)
        elif backbone == 'densenet201':
            from models.densenet_backbone import densenet201
            self.backbone = densenet201(pretrained=True)
            channels = (512, 1792, 1920)
        else:
            raise NotImplementedError

        # Extra layers

        self.ft_module = feature_transform_module(channels)
        self.pyramid_ext = pyramid_feature_extractor(self.fpn_level,
                channels)
        self.lateral_convs = lateral_convs(self.fpn_level)
        self.fpn_convs = fpn_convs(self.fpn_level)

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
        transformed_features = list()
        pyramid_fea = list()
        loc = list()
        conf = list()

        # backbone

        source_features = self.backbone(x)
        assert len(self.ft_module) == len(source_features)
        for (k, v) in zip(source_features, self.ft_module):
            transformed_features.append(v(k))
        x = torch.cat(transformed_features, 1)

        # detection

        for v in self.pyramid_ext:
            x = v(x)
            pyramid_fea.append(x)

        # fpn

        laterals = [lateral_conv(pyramid_fea[i]) for (i,
                    lateral_conv) in enumerate(self.lateral_convs)]
        for i in range(self.fpn_level - 1, 0, -1):
            size = laterals[i - 1].size()[-2:]
            laterals[i - 1] = laterals[i - 1] \
                + F.interpolate(laterals[i], size=size, mode='nearest')
        pyramid_fea = [self.fpn_convs[i](laterals[i]) for i in
                       range(self.fpn_level)]

        # apply multibox head to source layers

        for (i, (x, l, c)) in enumerate(zip(pyramid_fea, self.loc,
                self.conf)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0),
                -1, self.num_classes))  # loc preds
                                        # conf preds


def build_net(size=320, num_classes=20, backbone='vgg16'):

    if not size % 64 == 0:
        raise NotImplementedError('Error: Sorry size {} is not supported!'.format(size))

    return RetinaNet(size, num_classes, backbone)

