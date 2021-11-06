#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
from apex import amp
from data import detection_collate, DataPrefetcher
from utils import PriorBox
from utils import MultiBoxLoss
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
cudnn.benchmark = True

### For Reproducibility ###
# import random
# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.cuda.empty_cache()
# cudnn.benchmark = False
# cudnn.deterministic = True
# cudnn.enabled = True
### For Reproducibility ###

parser = argparse.ArgumentParser(description='Pytorch Training')
parser.add_argument('--neck', default='pafpn')
parser.add_argument('--backbone', default='resnet18')
parser.add_argument('--dataset', default='COCO')
parser.add_argument('--save_folder', default='weights/')
parser.add_argument('--multi_anchor', action='store_true')
parser.add_argument('--multi_level', action='store_true')
parser.add_argument('--mutual_guide', action='store_true')
parser.add_argument('--base_anchor_size', default=24.0, type=float)
parser.add_argument('--size', default=320, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--warm_iter', default=500, type=int)
args = parser.parse_args()
print(args)


if __name__ == '__main__':

    print('Loading Dataset...')
    if args.dataset == 'VOC':
        from data import VOCDetection
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(train_sets, args.size)
        epoch_size = len(dataset) // args.batch_size
        max_iter = 70 * epoch_size
    elif args.dataset == 'COCO':
        from data import COCODetection
        train_sets = [('2017', 'train')]
        dataset = COCODetection(train_sets, args.size)
        epoch_size = len(dataset) // args.batch_size
        max_iter = 140 * epoch_size
    elif args.dataset == 'XML':
        from data import XMLDetection
        dataset = XMLDetection('train', args.size)
        epoch_size = len(dataset) // args.batch_size
        max_iter = 100 * epoch_size
    else:
        raise NotImplementedError('Unkown dataset {}!'.format(args.dataset))

    print('Loading Optimizer & Network & Criterion...')
    from models.teacher_detector import Detector
    model = Detector(args.size, dataset.num_classes, args.backbone, args.neck,
        multi_anchor=args.multi_anchor, multi_level=args.multi_level).cuda()
    optimizer = optim.SGD(tencent_trick(model), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    ema_model = ModelEMA(model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total param is : {:e}'.format(num_param))

    print('Preparing Criterion & AnchorBoxes...')
    criterion = MultiBoxLoss(mutual_guide=args.mutual_guide)
    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size, 
        multi_anchor=args.multi_anchor, multi_level=args.multi_level).cuda()

    print('Training {}-{}-{} on {} with {} images'.format(
        'retina' if args.multi_anchor else 'fcos', 
        args.neck, args.backbone, dataset.name, len(dataset)),
    )
    os.makedirs(args.save_folder, exist_ok=True)
    timer = Timer()
    for iteration in range(max_iter):
        if iteration % epoch_size == 0:

            # create batch iterator
            rand_loader = data.DataLoader(
                dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=detection_collate
            )
            prefetcher = DataPrefetcher(rand_loader)
            model.train()

        # traning iteratoin
        timer.tic()
        adjust_learning_rate(optimizer, args.lr, iteration, args.warm_iter, max_iter)
        (images, targets) = prefetcher.next()

        # random resize
        if iteration < args.warm_iter or iteration >= 0.8*max_iter:
            new_size = args.size
        elif args.size == 320:
            new_size = 64 * (5 + random.choice([-1,0,1]))
        elif args.size == 512:
            new_size = 128 * (4 + random.choice([-1,0,1]))
        images = nn.functional.interpolate(
                images, size=(new_size, new_size), mode="bilinear", align_corners=False
            )
        priors = PriorBox(args.base_anchor_size, new_size, base_size=args.size, 
            multi_anchor=args.multi_anchor, multi_level=args.multi_level).cuda()

        out = model.forward_test(images)
        (loss_l, loss_c) = criterion(out[:2], priors, targets)
        loss = loss_l + loss_c

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        ema_model.update(model)
        load_time = timer.toc()

        # logging
        if iteration % 100 == 0:
            print('iter {}/{}, lr {:.6f}, loss_l {:.2f}, loss_c {:.2f}, loss {:.2f}, time {:.2f}s, eta {:.2f}h'.format(
                iteration,
                max_iter,
                optimizer.param_groups[0]['lr'],
                loss_l.item(),
                loss_c.item(),
                loss.item(),
                load_time,
                load_time * (max_iter - iteration) / 3600,
                ))
            timer.clear()

    # model saving
    model = ema_model.ema
    save_path = os.path.join(args.save_folder, '{}_{}_{}_{}_size{}_anchor{}{}.pth'.format(
        args.dataset,
        ('retina' if args.multi_anchor else 'fcos'),
        args.neck,
        args.backbone,
        args.size,
        args.base_anchor_size,
        ('_MG' if args.mutual_guide else ''),
        ))
    tosave = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
    print('Saving to {}'.format(save_path))
    torch.save(tosave, save_path)