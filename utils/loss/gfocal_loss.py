#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        loss_weight: float = 2.0,
    ) -> None:
        super(GFocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            pred, target = pred[mask], target[mask]
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        # focal_weight = (target - pred_sigmoid).abs().pow(self.gamma)
        focal_weight = (
            target * (target > 0.0).float()
            + self.alpha
            * (pred_sigmoid - target).abs().pow(self.gamma)
            * (target == 0.0).float()
        )
        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        loss = loss.sum() / (target > 0.0).float().sum()
        return loss * self.loss_weight
