#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from ..box import match, mutual_match, encode, decode
from .focal_loss import FocalLoss
from .gfocal_loss import GFocalLoss
from .balanced_l1_loss import BalancedL1Loss
from .giou_loss import GIOULoss


class MultiBoxLoss(nn.Module):
    """Object Detection Loss"""

    def __init__(
        self,
        mutual_guide: bool = True,
    ) -> None:
        super(MultiBoxLoss, self).__init__()

        self.mutual_guide = mutual_guide
        self.focal_loss = FocalLoss()
        self.gfocal_loss = GFocalLoss()
        self.l1_loss = BalancedL1Loss()
        self.iou_loss = GIOULoss()

    def forward(
        self,
        predictions: dict,
        priors: torch.Tensor,
        targets: list,
    ) -> tuple:
        (loc_p, cls_p) = predictions["loc"], predictions["conf"]
        (num, num_priors, num_classes) = cls_p.size()

        if self.mutual_guide:

            # match priors (default boxes) and ground truth boxes
            loc_t = torch.zeros(num, num_priors, 4).cuda()
            cls_t = torch.zeros(num, num_priors).cuda().long()
            cls_w = torch.zeros(num, num_priors).cuda()
            loc_w = torch.zeros(num, num_priors).cuda()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                regress = loc_p[idx, :, :]
                classif = cls_p[idx, :, :]
                mutual_match(
                    truths,
                    labels,
                    regress,
                    classif,
                    priors,
                    loc_t,
                    cls_t,
                    cls_w,
                    loc_w,
                    idx,
                )

            # Localization Loss
            pos = loc_w >= 3.0
            priors = priors.unsqueeze(0).expand_as(loc_p)
            mask = pos.unsqueeze(-1).expand_as(loc_p)

            weights = (loc_w - 3.0).relu().unsqueeze(-1).expand_as(loc_p)
            weights = weights[mask].view(-1, 4)
            weights = weights / weights.sum()

            loc_p = loc_p[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            loss_l = self.l1_loss(
                loc_p, encode(loc_t, priors), weights=weights
            ) + self.iou_loss(decode(loc_p, priors), loc_t, weights=weights.sum(-1))

            # Classification Loss
            cls_t = cls_t + 1
            neg = cls_w <= 1.0
            cls_t[neg] = 0
            cls_t = (
                torch.zeros(num * num_priors, num_classes + 1)
                .cuda()
                .scatter_(1, cls_t.view(-1, 1), 1)
            )
            cls_t = cls_t[:, 1:].view(
                num, num_priors, num_classes
            )  # shape: (batch_size, num_priors, num_classes)

            cls_w = (cls_w - 3.0).relu().unsqueeze(-1).expand_as(cls_t)
            loss_c = self.gfocal_loss(cls_p, cls_t * cls_w)

            return loss_l + loss_c

        else:

            # match priors (default boxes) and ground truth boxes
            cls_w = torch.zeros(num, num_priors).cuda()
            loc_t = torch.zeros(num, num_priors, 4).cuda()
            cls_t = torch.zeros(num, num_priors).cuda().long()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                match(truths, labels, priors, loc_t, cls_t, cls_w, idx)

            pos = cls_w >= 0.5
            ign = (cls_w < 0.5) * (cls_w >= 0.4)
            neg = cls_w < 0.4

            # Localization Loss
            priors = priors.unsqueeze(0).expand_as(loc_p)
            mask = pos.unsqueeze(-1).expand_as(loc_p)
            loc_p = loc_p[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loss_l = self.l1_loss(loc_p, encode(loc_t, priors))

            # Classification Loss
            cls_t[neg] = 0
            batch_label = (
                torch.zeros(num * num_priors, num_classes + 1)
                .cuda()
                .scatter_(1, cls_t.view(-1, 1), 1)
            )
            batch_label = batch_label[:, 1:].view(
                num, num_priors, num_classes
            )  # shape: (batch_size, num_priors, num_classes)
            ign = ign.unsqueeze(-1).expand_as(
                batch_label
            )  # shape: (batch_size, num_priors, num_classes)
            batch_label[ign] *= -1
            mask = batch_label >= 0
            loss_c = self.focal_loss(cls_p, batch_label, mask)

            return loss_l + loss_c
