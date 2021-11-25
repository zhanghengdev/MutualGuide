#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class GFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, loss_weight=2.0):
        super(GFocalLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred, target = pred[mask], target[mask]
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        focal_weight = (target - pred_sigmoid).abs().pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        loss = loss.sum() / (target>0.0).float().sum()
        return loss * self.loss_weight
