#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    def __init__(self, depth=18, pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        if pretrained:
            self.stem = None
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.stem = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                )            

        self.depth = depth
        if self.depth == 18:
            (block, layers) = (BasicBlock, [2, 2, 2, 2])
        elif self.depth == 34:
            (block, layers) = (BasicBlock, [3, 4, 6, 3])
        elif self.depth == 50:
            (block, layers) = (Bottleneck, [3, 4, 6, 3])
        elif self.depth == 101:
            (block, layers) = (Bottleneck, [3, 4, 23, 3])
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if pretrained:
            self.load_pre_trained_weights()

    def load_pre_trained_weights(self):
        print('Loading Pytorch pretrained weights...')
        pretrained_dict = {
            18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        }
        pretrained_dict = model_zoo.load_url(pretrained_dict[self.depth])
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
        self.load_state_dict(pretrained_dict, strict=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.layer3(x)
        out2 = self.layer4(out1)

        return out1, out2

