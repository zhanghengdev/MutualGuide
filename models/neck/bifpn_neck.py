#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_blocks import BasicConv

class CEM(nn.Module):

    """Context Enhancement Module"""

    def __init__(self, channels, fea_channel):
        super(CEM, self).__init__()
        self.cv1 = BasicConv(channels[0], fea_channel, kernel_size=1, padding=0)
        self.cv2 = BasicConv(channels[1], fea_channel, kernel_size=1, padding=0, scale_factor=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cv3 = BasicConv(channels[1], fea_channel, kernel_size=1, padding=0)

    def forward(self, inputs):
        C4_lat = self.cv1(inputs[0])
        C5_lat = self.cv2(inputs[1])
        Cglb_lat = self.cv3(self.gap(inputs[1]))
        return C4_lat + C5_lat + Cglb_lat


def fpn_feature_extractor(fpn_level, fea_channel):
    layers = [BasicConv(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1)]
    for _ in range(fpn_level - 1):
        layers.append(BasicConv(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)



class Swish(nn.Module):
    """
    Implement the Swish activation function.
    See: https://arxiv.org/abs/1710.05941 for more details.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

        
class BiFPNLayer(nn.Module):
    """
    This module implements one layer of BiFPN, and BiFPN can be obtained
    by stacking this module multiple times.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, input_size, in_channels_list, out_channels,
                 fuse_type="fast", norm="BN"):
        """
        input_size (int): the input image size.
        in_channels_list (list): the number of input tensor channels per level.
        out_channels (int): the number of output tensor channels.
        fuse_type (str): now only support three weighted fusion approaches:
            * fast:    Output = sum(Input_i * w_i / sum(w_j))
            * sotfmax: Output = sum(Input_i * e ^ w_i / sum(e ^ w_j))
            * sum:     Output = sum(Input_i) / len(Input_i)
        norm (str): the normalization to use.
        """
        super(BiFPNLayer, self).__init__()
        assert fuse_type in ("fast", "softmax", "sum"), f"Unknown fuse method: {fuse_type}." \
            " Please select in [fast, sotfmax, sum]."

        self.input_size = input_size
        self.in_channels_list = in_channels_list
        self.fuse_type = fuse_type
        self.levels = len(in_channels_list)
        self.nodes_input_offsets = [
            [3, 4],
            [2, 5],
            [1, 6],
            [0, 7],
            [1, 7, 8],
            [2, 6, 9],
            [3, 5, 10],
            [4, 11],
        ]
        self.nodes_strides = [
            2 ** x
            for x in [6, 5, 4, 3, 4, 5, 6, 7]
        ]

        # Change input feature map to have target number of channels.
        self.resample_convs = nn.ModuleList()
        for node_i_input_offsets in self.nodes_input_offsets:
            resample_convs_i = nn.ModuleList()
            for input_offset in node_i_input_offsets:
                if self.in_channels_list[input_offset] != out_channels:
                    resample_conv = Conv2d(
                        self.in_channels_list[input_offset],
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        norm=get_norm(norm, out_channels),
                        activation=None,
                    )
                else:
                    resample_conv = nn.Identity()
                self.in_channels_list.append(out_channels)
                resample_convs_i.append(resample_conv)
            self.resample_convs.append(resample_convs_i)

        # fpn combine weights
        self.edge_weights = nn.ParameterList()
        for node_i_input_offsets in self.nodes_input_offsets:
            # combine weight
            if fuse_type == "fast" or fuse_type == "softmax":
                weights_i = nn.Parameter(
                    torch.ones(len(node_i_input_offsets), dtype=torch.float32),
                    requires_grad=True,
                )
            elif fuse_type == "sum":
                weights_i = nn.Parameter(
                    torch.ones(len(node_i_input_offsets), dtype=torch.float32),
                    requires_grad=False,
                )
            else:
                raise ValueError("Unknown fuse method: {}".format(self.fuse_type))
            self.edge_weights.append(weights_i)

        # Convs for combine edge features
        self.combine_convs = nn.ModuleList()
        for node_i_input_offsets in self.nodes_input_offsets:
            combine_conv = SeparableConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding="SAME",
                norm=get_norm(norm, out_channels),
                activation=None,
            )
            self.combine_convs.append(combine_conv)

        self.act = Swish()
        self.down_sampling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up_sampling = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        print('start')
        assert len(inputs) == self.levels
        # Build top-down and bottom-up path
        self.nodes_features = inputs
        for node_idx, (node_i_input_offsets, node_i_stride) in enumerate(zip(self.nodes_input_offsets, self.nodes_strides)):
            # edge weights
            if self.fuse_type == "fast":
                weights_i = F.relu(self.edge_weights[node_idx])
            elif self.fuse_type == "softmax":
                weights_i = self.edge_weights[node_idx].softmax(dim=0)
            elif self.fuse_type == "sum":
                weights_i = self.edge_weights[node_idx]

            target_width = self.input_size / node_i_stride
            edge_features = []
            for offset_idx, offset in enumerate(node_i_input_offsets):
                edge_feature = self.nodes_features[offset]
                resample_conv = self.resample_convs[node_idx][offset_idx]
                # 1x1 conv for change feature map channels if necessary
                edge_feature = resample_conv(edge_feature)
                width = edge_feature.size(-1)
                if width > target_width:
                    # Downsampling for change feature map size
                    assert width / target_width == 2.0
                    edge_feature = self.down_sampling(edge_feature)
                elif width < target_width:
                    # Upsampling for change feature map size
                    assert target_width / width == 2.0
                    edge_feature = self.up_sampling(edge_feature)
                edge_feature = edge_feature * (weights_i[offset_idx] / (weights_i.sum() + 1e-4))
                edge_features.append(edge_feature)
            node_i_feature = sum(edge_features)
            node_i_feature = self.act(node_i_feature)
            node_i_feature = self.combine_convs[node_idx](node_i_feature)
            self.nodes_features.append(node_i_feature)

        # The number of node in one bifpn layer is 13
        assert len(self.nodes_features) == 13
        # The bifpn layer output is the last 5 nodes
        return self.nodes_features[-5:]



class BiFPNNeck(nn.Module):

    def __init__(self, fpn_level, channels, fea_channel):
        super(BiFPNNeck, self).__init__()
        self.fpn_level = fpn_level
        self.ft_module = CEM(channels, fea_channel)
        self.pyramid_ext = fpn_feature_extractor(self.fpn_level, fea_channel)
        self.lateral_convs = lateral_convs(self.fpn_level, fea_channel)
        self.fpn_convs = fpn_convs(self.fpn_level, fea_channel)
        self.downsample_convs = downsample_convs(self.fpn_level, fea_channel)
        self.pafpn_convs = fpn_convs(self.fpn_level, fea_channel)


    def forward(self, x):
        x = self.ft_module(x)
        fpn_fea = list()
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)
        laterals = [lateral_conv(x) for (x, lateral_conv) in zip(fpn_fea, self.lateral_convs)]
        for i in range(self.fpn_level - 1, 0, -1):
            size = laterals[i - 1].size()[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=size, mode='nearest')
        fpn_fea = [fpn_conv(x) for (x, fpn_conv) in zip(laterals, self.fpn_convs)]
        for i in range(0, self.fpn_level - 1):
            fpn_fea[i + 1] += self.downsample_convs[i](fpn_fea[i])
        pafpn_fea = [pafpn_conv(x) for (x, pafpn_conv) in zip(fpn_fea, self.pafpn_convs)]
        return pafpn_fea

