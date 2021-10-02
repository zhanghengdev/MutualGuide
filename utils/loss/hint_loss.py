#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box import get_foreground, get_foreground2
import numpy as np

class HintLoss(nn.Module):

    def __init__(self, mode='mse', factor=5.0, multi_anchor=True):
        super(HintLoss, self).__init__()
        self.mode = mode
        self.factor = factor
        self.multi_anchor = multi_anchor
        print('Using {} mode...'.format(self.mode))

    def forward(self, fea_t, fea_s, conf_t, conf_s, priors, targets, var=None):

        if self.mode == 'mse':
            return ((fea_s-fea_t)**2).mean() * self.factor / 2

        if self.mode == 'gauss':
            assert var is not None
            var = var.relu()
            loss =  torch.mean(
                        (fea_t - fea_s) ** 2 / (torch.exp(var) + 1e-6)
                        + var, dim=1
                    )
            return loss.mean() * self.factor

        elif self.mode == 'pad_l1':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1 - x2).abs()
                disagree = disagree.sum(-1).unsqueeze(1)
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1).expand_as(fea_t)
                weight = disagree / disagree.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor

        elif self.mode == 'pad_euc':
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

        elif self.mode == 'pad_kl':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                kl1 = x1 * torch.log(x1/x2)
                x1 = 1.0 - x1
                x2 = 1.0 - x2
                kl2 = x1 * torch.log(x1/x2)
                disagree = (kl1 + kl2).relu()
                disagree = disagree.sum(-1).unsqueeze(1)
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1).expand_as(fea_t)
                weight = disagree / disagree.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor
        
        elif self.mode == 'fg-only':
            with torch.no_grad():
                (num, num_priors, num_classes) = conf_t.size()
                mask = torch.zeros(num, num_priors, device='cuda:0')
                for idx in range(num):
                    truths = targets[idx][:, :-1]
                    get_foreground(truths, priors, mask, idx, self.multi_anchor)
                mask = Variable(mask, requires_grad=False)
                mask = mask.unsqueeze(1)
                if self.multi_anchor:
                    mask = F.avg_pool1d(mask, kernel_size=6, stride=6, padding=0)
                mask = mask.permute(0,2,1).expand_as(fea_t)
                weight = mask / mask.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor

        elif self.mode == 'decouple':
            with torch.no_grad():
                (num, num_priors, num_classes) = conf_t.size()
                mask = torch.zeros(num, num_priors, device='cuda:0')
                for idx in range(num):
                    truths = targets[idx][:, :-1]
                    get_foreground2(truths, priors, mask, idx)
                mask = Variable(mask, requires_grad=False)
                mask = mask.unsqueeze(1)
                if self.multi_anchor:
                    mask = F.avg_pool1d(mask, kernel_size=6, stride=6, padding=0)
                mask = mask.permute(0,2,1).expand_as(fea_t)
            return (mask*((fea_s-fea_t)**2)).mean() * 0.4 * self.factor + ((1.0-mask)*((fea_s-fea_t)**2)).mean() * 1.6 * self.factor

        elif self.mode == 'most-diff':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1-x2).abs()
                disagree = disagree.sum(-1).unsqueeze(1)
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1)
                weight = torch.zeros_like(disagree)
                weight[disagree > disagree.median()] = 1.0
                weight = weight.expand_as(fea_t)
                weight = weight / weight.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor

        elif self.mode == 'most-same':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1-x2).abs()
                disagree = disagree.sum(-1).unsqueeze(1)
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1)
                weight = torch.zeros_like(disagree)
                weight[disagree < disagree.median()] = 1.0
                weight = weight.expand_as(fea_t)
                weight = weight / weight.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor

        elif self.mode == 'most-diff2':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1-x2).abs()
                disagree = disagree.sum(-1).unsqueeze(1)
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1)
                weight = torch.zeros_like(disagree)
                thresh = disagree[disagree > disagree.median()].median()
                weight[disagree > thresh] = 1.0
                weight = weight.expand_as(fea_t)
                weight = weight / weight.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor

        elif self.mode == 'most-same2':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1-x2).abs()
                disagree = disagree.sum(-1).unsqueeze(1)
                if self.multi_anchor:
                    disagree = F.avg_pool1d(disagree, kernel_size=6, stride=6, padding=0)
                disagree = disagree.permute(0,2,1)
                weight = torch.zeros_like(disagree)
                thresh = disagree[disagree < disagree.median()].median()
                weight[disagree < thresh] = 1.0
                weight = weight.expand_as(fea_t)
                weight = weight / weight.sum()
            return (weight*((fea_s-fea_t)**2)).sum() * self.factor
