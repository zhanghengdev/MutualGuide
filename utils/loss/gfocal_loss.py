#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=1.5, loss_weight=2.0):
        super(GFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred, target = pred[mask], target[mask]

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)  # prob from logits
        binary_target = (target>0.0).float()
        alpha_factor = binary_target * self.alpha + (1 - binary_target) * (1 - self.alpha)
        modulating_factor = torch.abs(target - pred_prob) ** self.gamma
        loss = loss * alpha_factor * modulating_factor
        loss = loss.sum() / binary_target.sum()
        return loss * self.loss_weight
