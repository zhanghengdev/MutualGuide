#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetBackbone(nn.Module):
    def __init__(self, version="shufflenet-1.0", pretrained=True):
        super(ShuffleNetBackbone, self).__init__()
        self.version = version
        if self.version == "shufflenet-0.5":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
            self.out_channels = (96, 192)
        elif self.version == "shufflenet-1.0":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
            self.out_channels = (232, 464)
        elif self.version == "shufflenet-1.5":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
            self.out_channels = (352, 704)
        elif self.version == "shufflenet-2.0":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
            self.out_channels = (488, 976)
        else:
            raise ValueError

        input_channels = 3
        stages_repeats = [4, 8, 4]
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, stages_repeats, self._stage_out_channels[1:]
        ):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        if pretrained:
            self.load_pre_trained_weights()

            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

    def load_pre_trained_weights(self):
        print("Loading Pytorch pretrained weights...")
        if self.version == "shufflenet-0.5":
            state_dict = model_zoo.load_url(
                "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth"
            )
        elif self.version == "shufflenet-1.0":
            state_dict = model_zoo.load_url(
                "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth"
            )
        elif self.version == "shufflenet-1.5":
            state_dict = model_zoo.load_url(
                "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth"
            )
        elif self.version == "shufflenet-2.0":
            state_dict = model_zoo.load_url(
                "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth"
            )
        else:
            raise ValueError
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        out1 = self.stage3(x)
        out2 = self.stage4(out1)
        return out1, out2
