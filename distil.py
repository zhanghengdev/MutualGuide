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
import torch.utils.data as data
from data import detection_collate, DataPrefetcher
from utils import PriorBox
from utils import MultiBoxLoss, HintLoss
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
import torch.nn.functional as F
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
parser.add_argument('--mutual_guide', action='store_true')
parser.add_argument('--base_anchor_size', default=24.0, type=float)
parser.add_argument('--size', default=512, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--warm_iter', default=500, type=int)
parser.add_argument('--kd', default='pdf', help='Hint loss')
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
    
    print('Loading student Network...')
    from models.student_detector import Detector
    model = Detector(args.size, dataset.num_classes, args.backbone, args.neck).cuda()
    print(model)
    num_param = sum(p.numel() for p in model.parameters())
    print('Total trainable param of student model is : {:e}'.format(num_param))

    print('Loading teacher Network...')
    if args.backbone=='repvgg-A0':
        backbone = 'repvgg-A2'
    elif args.backbone=='repvgg-A1':
        backbone = 'repvgg-A2'
    elif args.backbone=='resnet18':
        backbone = 'resnet34'
    elif args.backbone=='regnet400':
        backbone = 'regnet800'
    elif args.backbone=='shufflenet-0.5':
        backbone = 'shufflenet-1.0'
    else:
        raise NotImplementedError
    neck = 'pafpn'
    from models.teacher_detector import Detector
    teacher = Detector(args.size, dataset.num_classes, backbone, neck).cuda()
    trained_model = 'weights/{}_retina_{}_{}_size{}_anchor{}_MG.pth'.format(
            args.dataset, neck, backbone, args.size, args.base_anchor_size,
    )
    print('loading weights from', trained_model)
    state_dict = torch.load(trained_model)
    teacher.load_state_dict(state_dict["model"], strict=True)
    teacher.deploy()
    num_param = sum(p.numel() for p in teacher.parameters())
    print('Total param of teacher model is : {:e}'.format(num_param))

    print('Preparing Optimizer & AnchorBoxes...')
    optimizer = optim.SGD(tencent_trick(model), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()
    ema_model = ModelEMA(model)
    criterion_det = MultiBoxLoss(mutual_guide=args.mutual_guide)
    criterion_kd = HintLoss(args.kd)
    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size).cuda()

    print('Training retina-{}-{} on {} with {} images'.format(
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
        if iteration >= 0.8*max_iter:
            new_size = args.size
        elif args.size == 320:
            new_size = 64 * (5 + random.choice([-1,0,1]))
        elif args.size == 512:
            new_size = 128 * (4 + random.choice([-1,0,1]))
        images = nn.functional.interpolate(
                images, size=(new_size, new_size), mode="bilinear", align_corners=False
            )
        priors = PriorBox(args.base_anchor_size, new_size, base_size=args.size).cuda()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                out_t = teacher(images)
            out = model(images)
            (loss_l, loss_c) = criterion_det(out, priors, targets)
            loss_kd = criterion_kd(out_t, out)
            loss = loss_l + loss_c + loss_kd
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema_model.update(model)
        load_time = timer.toc()

        # logging
        if iteration % 100 == 0:
            print('iter {}/{}, lr {:.6f}, loss_l {:.2f}, loss_c {:.2f}, loss_kd {:.2f}, loss {:.2f}, time {:.2f}s, eta {:.2f}h'.format(
                iteration,
                max_iter,
                optimizer.param_groups[0]['lr'],
                loss_l.item(),
                loss_c.item(),
                loss_kd.item(),
                loss.item(),
                load_time,
                load_time * (max_iter - iteration) / 3600,
                ))
            timer.clear()

    # model saving
    model = ema_model.ema
    save_path = os.path.join(args.save_folder, '{}_retina_{}_{}_size{}_anchor{}{}_kd{}.pth'.format(
        args.dataset,
        args.neck,
        args.backbone,
        args.size,
        args.base_anchor_size,
        ('_MG' if args.mutual_guide else ''),
        args.kd,
        ))
    tosave = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
    print('Saving to {}'.format(save_path))
    torch.save(tosave, save_path)
    
