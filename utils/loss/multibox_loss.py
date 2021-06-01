#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box import match, mutual_match, encode, decode
from .balanced_l1_loss import BalancedL1Loss
from .focal_loss import FocalLoss


class MultiBoxLoss(nn.Module):

    def __init__(self, mutual_guide):
        super(MultiBoxLoss, self).__init__()
        self.mutual_guide = mutual_guide
        self.focal_loss = FocalLoss(alpha=0.25, gamma=1.0)
        self.reg_loss = BalancedL1Loss(alpha=0.5, gamma=1.5, beta=0.11)

    def forward(self, predictions, priors, targets):
        (loc_data, conf_data) = predictions
        (num, num_priors, num_classes) = conf_data.size()

        if self.mutual_guide:

            # match priors (default boxes) and ground truth boxes
            loc_t = torch.zeros(num, num_priors, 4, device='cuda:0')
            conf_t = torch.zeros(num, num_priors, device='cuda:0').long()
            overlap_t = torch.zeros(num, num_priors, device='cuda:0')
            pred_t = torch.zeros(num, num_priors, device='cuda:0')
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                regress = loc_data[idx, :, :]
                classif = conf_data[idx, :, :]
                mutual_match(truths, priors, regress, classif, labels, loc_t, conf_t, overlap_t, pred_t, idx)
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)
            overlap_t = Variable(overlap_t, requires_grad=False)
            pred_t = Variable(pred_t, requires_grad=False)

            # Localization Loss (Smooth L1)
            pos = pred_t == 3.0
            priors = priors.unsqueeze(0).expand_as(loc_data)
            mask = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loc_t = encode(loc_t, priors)
            loss_l = self.reg_loss(loc_p, loc_t, reduction='mean')

            # Classification Loss
            pos = overlap_t == 3.0
            ign = overlap_t == 2.0
            neg = overlap_t <= 1.0
            conf_t[neg] = 0
            with torch.no_grad():
                batch_label = torch.zeros(num * num_priors, num_classes + 1, device='cuda:0').scatter_(1, conf_t.view(-1, 1), 1)
                batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)  # shape: (batch_size, num_priors, num_classes)
                ign = ign.unsqueeze(-1).expand_as(batch_label)  # shape: (batch_size, num_priors, num_classes)
                batch_label[ign] *= -1
                mask = batch_label >= 0
            loss_c = self.focal_loss(conf_data[mask], batch_label[mask], reduction='mean')

            return (loss_l, loss_c)

        else:

            # match priors (default boxes) and ground truth boxes
            overlap_t = torch.zeros(num, num_priors, device='cuda:0')
            loc_t = torch.zeros(num, num_priors, 4, device='cuda:0')
            conf_t = torch.zeros(num, num_priors, device='cuda:0').long()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                match(truths, priors, labels, loc_t, conf_t, overlap_t, idx)
            overlap_t = Variable(overlap_t, requires_grad=False)
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)

            pos = overlap_t >= 0.5
            ign = (overlap_t < 0.5) * (overlap_t >= 0.4)
            neg = overlap_t < 0.4

            # Localization Loss (Smooth L1)
            priors = priors.unsqueeze(0).expand_as(loc_data)
            mask = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loc_t = encode(loc_t, priors)
            loss_l = self.reg_loss(loc_p, loc_t, reduction='mean')

            # Classification Loss
            conf_t[neg] = 0
            with torch.no_grad():
                batch_label = torch.zeros(num * num_priors, num_classes + 1, device='cuda:0').scatter_(1, conf_t.view(-1, 1), 1)
                batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)  # shape: (batch_size, num_priors, num_classes)
                ign = ign.unsqueeze(-1).expand_as(batch_label)  # shape: (batch_size, num_priors, num_classes)
                batch_label[ign] *= -1
                mask = batch_label >= 0
            loss_c = self.focal_loss(conf_data[mask], batch_label[mask], reduction='mean')

            return (loss_l, loss_c)