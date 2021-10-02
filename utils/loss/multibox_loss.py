#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box import match, mutual_match, encode, decode
from .focal_loss import FocalLoss
from .balanced_l1_loss import BalancedL1Loss
from .giou_loss import GIOULoss
from .ghm_loss import GHMLoss
from .hnm_loss import HNMLoss
from .gfocal_loss import GFocalLoss

class MultiBoxLoss(nn.Module):

    def __init__(self, mutual_guide=True, multi_anchor=True):
        super(MultiBoxLoss, self).__init__()
        self.mutual_guide = mutual_guide
        self.multi_anchor = multi_anchor
        
        self.hnm_loss = HNMLoss()
        self.focal_loss = FocalLoss()
        self.gfocal_loss = GFocalLoss()
        self.ghm_loss = GHMLoss()
        
        self.l1_loss = BalancedL1Loss()
        self.iou_loss = GIOULoss()


    def forward(self, predictions, priors, targets):
        (loc_data, conf_data) = predictions
        (num, num_priors, num_classes) = conf_data.size()

        if self.mutual_guide:

            # match priors (default boxes) and ground truth boxes
            with torch.no_grad():
                loc_t = torch.zeros(num, num_priors, 4).cuda()
                conf_t = torch.zeros(num, num_priors).cuda().long()
                overlap_t = torch.zeros(num, num_priors).cuda()
                pred_t = torch.zeros(num, num_priors).cuda()
                for idx in range(num):
                    truths = targets[idx][:, :-1]
                    labels = targets[idx][:, -1].long()
                    regress = loc_data[idx, :, :]
                    classif = conf_data[idx, :, :]
                    mutual_match(truths, priors, regress, classif, labels, loc_t, conf_t, overlap_t, pred_t, idx, self.multi_anchor)
                loc_t = Variable(loc_t, requires_grad=False)
                conf_t = Variable(conf_t, requires_grad=False)
                overlap_t = Variable(overlap_t, requires_grad=False)
                pred_t = Variable(pred_t, requires_grad=False)

            # Localization Loss (Smooth L1)
            pos = pred_t >= 3.0
            priors = priors.unsqueeze(0).expand_as(loc_data)
            mask = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loss_l = self.l1_loss(loc_p, encode(loc_t, priors))

            # Classification Loss
            neg = overlap_t <= 1.0
            conf_t[neg] = 0
            with torch.no_grad():
                batch_label = torch.zeros(num * num_priors, num_classes + 1).cuda().scatter_(1, conf_t.view(-1, 1), 1)
                batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)  # shape: (batch_size, num_priors, num_classes)
                score = (overlap_t-3.0).relu().unsqueeze(-1).expand_as(batch_label)
                batch_label = batch_label * score
                mask = batch_label >= 0
            loss_c = self.gfocal_loss(conf_data, batch_label, mask)
            return (loss_l, loss_c)

        else:

            # match priors (default boxes) and ground truth boxes
            overlap_t = torch.zeros(num, num_priors).cuda()
            loc_t = torch.zeros(num, num_priors, 4).cuda()
            conf_t = torch.zeros(num, num_priors).cuda().long()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                match(truths, priors, labels, loc_t, conf_t, overlap_t, idx, self.multi_anchor)
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
            loss_l = self.l1_loss(loc_p, encode(loc_t, priors))

            # Classification Loss
            conf_t[neg] = 0
            with torch.no_grad():
                batch_label = torch.zeros(num * num_priors, num_classes + 1).cuda().scatter_(1, conf_t.view(-1, 1), 1)
                batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)  # shape: (batch_size, num_priors, num_classes)
                ign = ign.unsqueeze(-1).expand_as(batch_label)  # shape: (batch_size, num_priors, num_classes)
                batch_label[ign] *= -1
                mask = batch_label >= 0
            loss_c = self.focal_loss(conf_data, batch_label, mask)

            return (loss_l, loss_c)
