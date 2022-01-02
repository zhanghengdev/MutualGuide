#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class BalancedL1Loss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.5,
        beta: float = 0.11,
    ) -> None:
        super(BalancedL1Loss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        b = np.e ** (self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta, self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff, self.gamma * diff + self.gamma / b - self.alpha * self.beta)
        if weights is None:
            return loss.mean()
        else:
            return (loss * weights).sum()

