#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class BalancedL1Loss(nn.Module):

    def __init__(self, alpha=0.5, gamma=1.5, beta=0.11):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred, target, reduction='mean'):
        diff = torch.abs(pred - target)
        b = np.e ** (self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta, self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff, self.gamma * diff + self.gamma / b - self.alpha * self.beta)
        if reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        else:
            return loss

