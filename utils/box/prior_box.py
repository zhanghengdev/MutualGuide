#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import math
from math import sqrt as sqrt
from itertools import product as product


def get_prior_box(
    anchor_size: float,
    image_size: int,
    scales: list = (1, sqrt(2)),
    aspect_ratios: list = (1.0, 0.5, 2.0),
) -> torch.Tensor:
    """Predefined anchor boxes"""

    repeat = 3 if image_size <= 640 else 4
    feature_size = [math.ceil(image_size / 2 ** (4 + i)) for i in range(repeat)]

    scales = [s * anchor_size / image_size for s in scales]
    scales = torch.as_tensor(scales)
    aspect_ratios = torch.as_tensor(aspect_ratios)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios
    ws = (w_ratios[None, :] * scales[:, None]).view(-1)
    hs = (h_ratios[None, :] * scales[:, None]).view(-1)
    cell_anchors = torch.stack(
        [torch.zeros_like(ws), torch.zeros_like(hs), ws, hs], dim=1
    )

    output = []
    for (k, f) in enumerate(feature_size):

        grid_width = grid_height = f
        shifts_x = (torch.arange(0, grid_width) + 0.5) / grid_width
        shifts_y = (torch.arange(0, grid_height) + 0.5) / grid_height
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack(
            (shift_x, shift_y, torch.zeros_like(shift_x), torch.zeros_like(shift_y)),
            dim=1,
        )

        level_cell_anchors = cell_anchors.clone() * 2**k
        level_anchors = (
            shifts.view(-1, 1, 4) + level_cell_anchors.view(1, -1, 4)
        ).view(-1, 4)
        output.append(level_anchors)

    output = torch.cat(output, 0)
    output.clamp_(max=1, min=0)
    return output
