#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred,
        target,
        reduction='sum',
        ):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        if reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.sum() / (target==1.0).float().sum()
        else:
            return loss
