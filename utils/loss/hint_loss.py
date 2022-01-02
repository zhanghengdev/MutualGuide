#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class HintLoss(nn.Module):

    def __init__(
        self,
        mode: str = 'pdf',
        loss_weight: float = 5.0,
    ) -> None:
        super(HintLoss, self).__init__()

        self.mode = mode
        self.loss_weight = loss_weight
        print('Using {} mode...'.format(self.mode))

    def forward(
        self,
        pred_t: torch.Tensor,
        pred_s: torch.Tensor,
    ) -> torch.Tensor:
        loc_t, conf_t, fea_t = pred_t['loc'], pred_t['conf'], pred_t['feature']
        loc_t, conf_t, fea_t = loc_t.detach(), conf_t.detach(), fea_t.detach()
        loc_s, conf_s, fea_s = pred_s['loc'], pred_s['conf'], pred_s['feature']

        if self.mode == 'mse':
            return ((fea_s-fea_t)**2).mean() * self.loss_weight

        if self.mode == 'pdf':
            with torch.no_grad():
                x1 = conf_t.sigmoid()
                x2 = conf_s.sigmoid()
                disagree = (x1 - x2) ** 2
                weight = disagree.sum(-1).unsqueeze(1).sqrt()
                weight = F.avg_pool1d(weight, kernel_size=6, stride=6, padding=0)  # 6 anchor per location
                weight = weight.permute(0,2,1).expand_as(fea_t)
                weight = weight / weight.sum()
            loss_pdf = (weight*((fea_s-fea_t)**2)).sum() * self.loss_weight
                    
            loss_cls = F.binary_cross_entropy_with_logits(conf_s, x1, reduction='none') * disagree
            loss_cls = loss_cls.sum() / (x1>0.5).float().sum()

            loss_reg = F.l1_loss(loc_s, loc_t)

            return loss_pdf + loss_cls + loss_reg
            
        raise NotImplementedError
