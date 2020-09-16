#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):

    """Predefined anchor boxes"""

    def __init__(self, base_anchor, image):
        super(PriorBox, self).__init__()
        repeat = (4 if image < 512 else 5)
        self.image = int(image)
        self.base_anchor = base_anchor
        self.feature_map = [math.ceil(self.image / 2 ** (3 + i))
                            for i in range(repeat)]

    def forward(self):
        mean = []
        for (k, (f_h, f_w)) in enumerate(zip(self.feature_map,
                self.feature_map)):
            for (i, j) in product(range(f_h), range(f_w)):
                cy = (i + 0.5) / f_h
                cx = (j + 0.5) / f_w

                anchor = self.base_anchor * 2 ** k / self.image
                mean += [cx, cy, anchor, anchor]
                mean += [cx, cy, anchor * sqrt(2), anchor / sqrt(2)]
                mean += [cx, cy, anchor / sqrt(2), anchor * sqrt(2)]
                anchor *= sqrt(2)
                mean += [cx, cy, anchor, anchor]
                mean += [cx, cy, anchor * sqrt(2), anchor / sqrt(2)]
                mean += [cx, cy, anchor / sqrt(2), anchor * sqrt(2)]

        # back to torch land

        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output

