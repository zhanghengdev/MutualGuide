#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class HintLoss(nn.Module):

    def __init__(self, mode='pdf', factor=5.0, multi_anchor=True):
        super(HintLoss, self).__init__()
        self.mode = mode
        self.factor = factor
        self.multi_anchor = multi_anchor
        print('Using {} mode...'.format(self.mode))

    def forward(self, fea_t, fea_s, conf_t, conf_s, priors, targets, var=None):

        if self.mode == 'mse':
            return ((fea_s-fea_t)**2).mean() * self.factor

        if self.mode == 'pdf':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1 - x2) ** 2
                disagree = disagree.sum(-1).unsqueeze(1).sqrt()
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1).expand_as(fea_t)
                weight = disagree / disagree.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor

        raise NotImplementedError