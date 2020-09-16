#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from .box_utils import decode


class Detect(Function):

    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        (loc, conf) = predictions
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        assert loc_data.size(0) == 1, \
            'Batch size = {} during evaluation'.format(loc_data.size(0))
        decoded_boxes = decode(loc_data.squeeze(0),
                               prior_data).clamp(min=0, max=1)
        conf_scores = conf_data.squeeze(0).sigmoid()
        return (decoded_boxes, conf_scores)

