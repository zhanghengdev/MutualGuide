#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


class VGGBackbone(nn.Module):
    def __init__(self, version="vgg11", pretrained=True):
        super(VGGBackbone, self).__init__()
        if version == "vgg11":
            cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
            dict_url = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"
            break_layer = 21
        elif version == "vgg16":
            cfg = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
            ]
            dict_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
            break_layer = 33
        else:
            raise ValueError

        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        self.layer1 = nn.Sequential(*layers[:break_layer])
        self.layer2 = nn.Sequential(*layers[break_layer:])

        if pretrained:
            print("Loading Pytorch pretrained weights...")
            pretrained_dict = model_zoo.load_url(dict_url)
            pretrained_dict = {
                k.replace("features.", "", 1): v
                for k, v in pretrained_dict.items()
                if "features" in k
            }
            self.layer1.load_state_dict(
                {
                    k: v
                    for k, v in pretrained_dict.items()
                    if int(k.split(".")[0]) < break_layer
                }
            )
            self.layer2.load_state_dict(
                {
                    self._rename(k, break_layer): v
                    for k, v in pretrained_dict.items()
                    if int(k.split(".")[0]) >= break_layer
                }
            )

    def _rename(self, k, num):
        a = int(k.split(".")[0])
        return k.replace("{}.".format(a), "{}.".format(a - num), 1)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out1, out2
