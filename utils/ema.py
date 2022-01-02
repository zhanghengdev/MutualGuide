import math
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    """ Model Exponential Moving Average """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        updates: int = 0,
    ) -> None:
        
        self.ema = deepcopy(model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(
        self,
        model: nn.Module,
    ) -> None:
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(
        self,
        model: nn.Module,
        include: tuple = (),
        exclude: tuple = ("process_group", "reducer"),
    ) -> None:
        # Update EMA attributes
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)