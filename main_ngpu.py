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
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
from data import detection_collate, preproc_for_test
from utils import PriorBox, Detect
from utils import MultiBoxLoss
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
from utils.box import SeqBoxMatcher
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
parser.add_argument('--save_folder', default='weights/public/')
parser.add_argument('--multi_anchor', action='store_true')
parser.add_argument('--multi_level', action='store_true')
parser.add_argument('--mutual_guide', action='store_true')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--seq_matcher', action='store_true')
parser.add_argument('--base_anchor_size', default=24.0, type=float)
parser.add_argument('--size', default=320, type=int)
parser.add_argument('--eval_thresh', default=0.05, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--warm_iter', default=500, type=int)
parser.add_argument('--trained_model', help='Location to trained model')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--draw', action='store_true', help='Draw detection results')
args = parser.parse_args()
args.warm_iter = args.warm_iter // args.ngpu
print(args)


def load_dataset():
    if args.dataset == 'VOC':
        from data import VOCDetection
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(train_sets, args.size)
        epoch_size = len(dataset) // args.batch_size // args.ngpu
        max_iter = 70 * epoch_size
        testset = VOCDetection([('2007', 'test')], args.size)
    elif args.dataset == 'COCO':
        from data import COCODetection
        train_sets = [('2017', 'train')]
        dataset = COCODetection(train_sets, args.size)
        epoch_size = len(dataset) // args.batch_size // args.ngpu
        max_iter = 150 * epoch_size
        testset = COCODetection([('2017', 'val')], args.size)
    elif args.dataset == 'XML':
        from data import XMLDetection
        dataset = XMLDetection('train', args.size)
        epoch_size = len(dataset) // args.batch_size // args.ngpu
        max_iter = 100 * epoch_size
        testset = XMLDetection('val', args.size)
    else:
        raise NotImplementedError('Unkown dataset {}!'.format(args.dataset))
    return (dataset, epoch_size, max_iter, testset)


def save_weights(model, optimizer):
    
    if args.local_rank != 0:
        return
    
    save_path = os.path.join(args.save_folder, '{}_{}_{}_{}_size{}_anchor{}{}_GFocal1011_ngpu.pth'.format(
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


if __name__ == '__main__':

    print('Prepare multi GPU training...')
    print('This precess in on devie GPU-{}'.format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device('cuda:{}'.format(args.local_rank))

    print('Loading Dataset...')
    (dataset, epoch_size, max_iter, testset) = load_dataset()

    print('Loading Optimizer & Network...')
    from models.teacher_detector import Detector
    model = Detector(args.size, dataset.num_classes, args.backbone, args.neck,
        multi_anchor=args.multi_anchor, multi_level=args.multi_level, pretrained=args.pretrained)
    model = convert_syncbn_model(model)
    model = model.to(device)
    optimizer = optim.SGD(tencent_trick(model), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # ema_model = ModelEMA(model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total param is : {:e}'.format(num_param))

    print('Preparing Criterion & AnchorBoxes...')
    criterion = MultiBoxLoss(mutual_guide=args.mutual_guide, multi_anchor=args.multi_anchor).to(device)
    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size, 
        multi_anchor=args.multi_anchor, multi_level=args.multi_level).to(device)

    if args.trained_model is not None:
        print('Loading weights from', args.trained_model)
        state_dict = torch.load(args.trained_model)
        model.load_state_dict(state_dict["model"], strict=True)
    else:
        print('Training {}-{}-{} on {} with {} images'.format(
            'retina' if args.multi_anchor else 'fcos', 
            args.neck, args.backbone, dataset.name, len(dataset)),
        )
        os.makedirs(args.save_folder, exist_ok=True)
        timer = Timer()
        train_sampler  = torch.utils.data.distributed.DistributedSampler(dataset)
        for iteration in range(max_iter):
            if iteration % epoch_size == 0:

                # save model
                save_weights(model, optimizer)
                
                # random resize
                new_size = args.size
                print('Switching image size to {}...'.format(new_size))
                dataset.size = new_size
                priors = PriorBox(args.base_anchor_size, new_size, base_size=args.size, 
                    multi_anchor=args.multi_anchor, multi_level=args.multi_level).to(device)

                # create batch iterator
                rand_loader = data.DataLoader(
                    dataset, args.batch_size, shuffle=False, num_workers=4, collate_fn=detection_collate,
                    pin_memory=False, drop_last=True, sampler=train_sampler,
                )
                batch_iterator = iter(rand_loader)
                # ema_model.update_attr(model)
                model.train()

            # traning iteratoin
            timer.tic()
            adjust_learning_rate(optimizer, args.lr, iteration, args.warm_iter, max_iter)
            (images, targets) = next(batch_iterator)
            images = Variable(images.to(device))
            targets = [Variable(anno.to(device)) for anno in targets]
            out = model.forward_test(images)
            (loss_l, loss_c) = criterion(out[:2], priors, targets)
            loss = loss_l + loss_c
            optimizer.zero_grad()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            # ema_model.update(model)
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
        # model = ema_model.ema
        save_weights(model, optimizer)
