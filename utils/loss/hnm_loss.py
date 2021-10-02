#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class HNMLoss(nn.Module):

    def __init__(self, ratio=3.0, loss_weight=1.0):
        super(HNMLoss, self).__init__()
        self.ratio = ratio
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask, reduction='mean'):
        pred, target = pred[mask], target[mask]
        with torch.no_grad():
            num_pos = target.sum().item()
            pt = pred.sigmoid() * (1 - target) + 2.0 * target
            mask = torch.topk(pt, int((1+self.ratio)*num_pos))[1]
        loss = F.binary_cross_entropy_with_logits(pred[mask], target[mask], reduction='none')
        if reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.sum() / num_pos
        else:
            return loss
