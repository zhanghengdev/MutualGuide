#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class HintLoss(nn.Module):
    def __init__(
        self,
        mode: str = "pdf",
        loss_weight: float = 5.0,
    ) -> None:
        super(HintLoss, self).__init__()

        self.mode = mode
        self.loss_weight = loss_weight
        print("INFO: Using {} mode...".format(self.mode))

    def forward(
        self,
        pred_t: torch.Tensor,
        pred_s: torch.Tensor,
    ) -> torch.Tensor:
        conf_t, fea_t = pred_t["conf"].detach(), pred_t["feature"].detach()
        conf_s, fea_s = pred_s["conf"].detach(), pred_s["feature"]

        if self.mode == "mse":
            return ((fea_s - fea_t) ** 2).mean() * self.loss_weight

        if self.mode == "pdf":

            with torch.no_grad():
                disagree = (conf_t.sigmoid() - conf_s.sigmoid()) ** 2
                weight = disagree.mean(-1).unsqueeze(1)
                weight = F.avg_pool1d(
                    weight, kernel_size=6, stride=6, padding=0
                )  # 6 anchor per location
                weight = weight.squeeze() / weight.sum()

            return (weight * ((fea_s - fea_t) ** 2).mean(-1)).sum() * self.loss_weight

        raise NotImplementedError
