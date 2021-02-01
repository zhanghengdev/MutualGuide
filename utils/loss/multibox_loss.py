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

    def __init__(self, num_classes, mutual_guide):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes - 1
        self.mutual_guide = mutual_guide
        self.focal_loss = FocalLoss(alpha=0.25, gamma=1.0)
        self.reg_loss = BalancedL1Loss(alpha=0.5, gamma=1.5, beta=0.11)

    def forward(self, predictions, priors, targets):
        (loc_data, conf_data) = predictions
        num = loc_data.size(0)  # loc_data should be (batch_size,num_priors,4)
        num_priors = priors.size(0)  # priors should be (num_priors,4)

        if self.mutual_guide:

            # match priors (default boxes) and ground truth boxes
            loc_t = torch.zeros(num, num_priors, 4).cuda()
            conf_t = torch.LongTensor(num, num_priors).cuda()
            overlap_t = torch.zeros(num, num_priors).cuda()
            pred_t = torch.zeros(num, num_priors).cuda()
            defaults = priors.data
            for idx in range(num):
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                labels = labels.long()
                regress = loc_data[idx, :, :].data
                classif = conf_data[idx, :, :].data
                mutual_match(truths, defaults, regress, classif, labels, loc_t, conf_t, overlap_t, pred_t, idx)
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)
            overlap_t = Variable(overlap_t, requires_grad=False)
            pred_t = Variable(pred_t, requires_grad=False)

            # Localization Loss (Smooth L1)
            pos = pred_t >= 3.0
            pos_reg_num = max(pos.data.float().sum(), 1)
            pos_idx = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            defaults = defaults.unsqueeze(0).expand_as(loc_data)
            defaults = defaults[pos_idx].view(-1, 4)
            loc_t = encode(loc_t, defaults)
            loss_l = self.reg_loss(loc_p, loc_t, reduction='sum')
            loss_l /= pos_reg_num * 4

            # Classification Loss
            pos = overlap_t >= 3.0
            ign = (overlap_t >= 2.0) * (overlap_t < 3.0)
            neg = overlap_t < 2.0
            pos_cls_num = max(pos.data.float().sum(), 1)
            conf_t[(ign + neg).gt(0)] = 0
            with torch.no_grad():
                conf_label = torch.FloatTensor(num * num_priors, self.num_classes + 1).cuda().zero_()
                conf_label.scatter_(1, conf_t.view(-1, 1), 1)
                conf_label = conf_label[:, 1:]
            pos_idx = (pos + neg).gt(0).unsqueeze(-1).expand_as(conf_data)
            conf_data = conf_data[pos_idx].view(-1, self.num_classes)
            conf_label = conf_label[pos_idx.view(-1, self.num_classes)].view(-1, self.num_classes)
            loss_c = self.focal_loss(conf_data, conf_label, reduction='sum')
            loss_c /= pos_cls_num

            return (loss_l, loss_c)
        else:

            # match priors (default boxes) and ground truth boxes
            overlap_t = torch.Tensor(num, num_priors).cuda()
            loc_t = torch.Tensor(num, num_priors, 4).cuda()
            conf_t = torch.LongTensor(num, num_priors).cuda()
            defaults = priors.data
            obj_num = 0
            for idx in range(num):
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                match(truths, defaults, labels, loc_t, conf_t, overlap_t, idx)
                obj_num += len(labels)
            overlap_t = Variable(overlap_t, requires_grad=False)
            loc_t = Variable(loc_t, requires_grad=False)
            conf_t = Variable(conf_t, requires_grad=False)

            pos = overlap_t >= 0.5
            ign = (overlap_t < 0.5) * (overlap_t >= 0.4)
            neg = overlap_t < 0.4
            pos_num = max(pos.data.float().sum(), 1)

            # Localization Loss (Smooth L1)
            pos_idx = pos.unsqueeze(-1).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            defaults = defaults.unsqueeze(0).expand_as(loc_data)
            defaults = defaults[pos_idx].view(-1, 4)
            loc_t = encode(loc_t, defaults)
            loss_l = self.reg_loss(loc_p, loc_t, reduction='sum')
            loss_l /= pos_num * 4

            # Classification Loss
            conf_t[(ign + neg).gt(0)] = 0
            posneg_idx = (pos + neg).gt(0).unsqueeze(-1).expand_as(conf_data)
            conf_data = conf_data[posneg_idx].view(-1, self.num_classes)
            with torch.no_grad():
                batch_label = torch.FloatTensor(num * num_priors, self.num_classes + 1).cuda().zero_()
                batch_label.scatter_(1, conf_t.view(-1, 1), 1)
                batch_label = batch_label[:, 1:][posneg_idx.view(-1, self.num_classes)].view(-1, self.num_classes)
            loss_c = self.focal_loss(conf_data, batch_label, reduction='sum')
            loss_c /= pos_num

            return (loss_l, loss_c)

