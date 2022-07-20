#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torchvision
import numpy as np
from .box_utils import decode


def post_process(
    predictions: torch.Tensor,
    prior: torch.Tensor,
    scale: torch.Tensor,
    eval_thresh: float = 0.05,
    nms_thresh: float = 0.5,
) -> tuple:
    """Detect layer at test time"""

    (loc, conf) = predictions["loc"], predictions["conf"]
    assert loc.size(0) == 1, "ERROR: Batch size = {} during evaluation".format(
        loc.size(0)
    )
    num_classes = conf.size(1)

    (loc, conf) = loc.squeeze(0), conf.squeeze(0)
    loc = decode(loc, prior).clamp(min=0, max=1)
    loc *= scale  # scale each detection back up to the image
    conf = conf.sigmoid()

    keep = conf.max(1)[0] > eval_thresh
    if not keep.any():
        return (np.empty([0, 4]), np.empty([0, num_classes]))
    loc = loc[keep]
    conf = conf[keep]

    keep = torchvision.ops.nms(loc, conf.max(1)[0], iou_threshold=nms_thresh)
    loc = loc[keep]
    conf = conf[keep]

    (loc, conf) = loc.cpu().numpy(), conf.cpu().numpy()
    return (loc, conf)
