#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride,
        groups=1,
        bias=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def switch_to_deploy(
        self,
    ) -> None:
        fusedconv = (
            nn.Conv2d(
                self.conv.in_channels,
                self.conv.out_channels,
                kernel_size=self.conv.kernel_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                groups=self.conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(self.conv.weight.device)
        )

        # prepare filters
        w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
        w_bn = torch.diag(
            self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var))
        )
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = (
            torch.zeros(self.conv.weight.size(0), device=self.conv.weight.device)
            if self.conv.bias is None
            else self.conv.bias
        )
        b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
            torch.sqrt(self.bn.running_var + self.bn.eps)
        )
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        self.conv = fusedconv  # update conv
        delattr(self, "bn")  # remove batchnorm
        self.forward = self.fuseforward  # update forward


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarkNetBackbone(nn.Module):
    def __init__(self, version="cspdarknet-0.5", pretrained=True):
        super(CSPDarkNetBackbone, self).__init__()

        if version == "cspdarknet-0.5":
            dep_mul, wid_mul = 0.33, 0.5
            self.out_channels = (256, 512)
        elif version == "cspdarknet-0.75":
            dep_mul, wid_mul = 0.67, 0.75
            self.out_channels = (384, 768)
        elif version == "cspdarknet-1.0":
            dep_mul, wid_mul = 1.0, 1.0
            self.out_channels = (512, 1024)
        else:
            raise ValueError

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3)

        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
            ),
        )

        if pretrained:
            print("Loading Pytorch pretrained weights...")
            pretrained_dict = torch.load(
                "weights/TorchPretrained/yolox_{}.pth".format(version)
            )["model"]
            keys = list(pretrained_dict.keys())
            for k in keys:
                if k.startswith("backbone.backbone."):
                    pretrained_dict[k[18:]] = pretrained_dict[k]
                    pretrained_dict.pop(k)
                else:
                    pretrained_dict.pop(k)
            self.load_state_dict(pretrained_dict, strict=True)

            # freeze params
            for module in [self.stem, self.dark2]:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        out1 = self.dark4(x)
        out2 = self.dark5(out1)
        return out1, out2
