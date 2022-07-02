#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn


def adjust_learning_rate(
    optimizer: torch.optim,
    base_lr: float,
    iteration: int,
    warm_iter: int,
    max_iter: int,
    min_lr_ratio: float = 0.05,
) -> float:
    """warmup + cosine lr decay"""
    start_lr = base_lr * min_lr_ratio
    if iteration <= warm_iter:
        lr = start_lr + (base_lr - start_lr) * iteration / warm_iter
    else:
        lr = start_lr + (base_lr - start_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (iteration - warm_iter) / (max_iter - warm_iter))
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def tencent_trick(
    model: nn.Module,
) -> list:
    """no weight decay for bn and conv bias"""
    (decay, no_decay) = ([], [])
    for (name, param) in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]
