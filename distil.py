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
from data import detection_collate, preproc_for_test
from utils import PriorBox, Detect
from utils import MultiBoxLoss, HintLoss
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
parser.add_argument('--neck', default='fpn')
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
parser.add_argument('--eval_thresh', default=0.01, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--warm_iter', default=500, type=int)
parser.add_argument('--trained_model', help='Location to trained model')
parser.add_argument('--draw', action='store_true', help='Draw detection results')
parser.add_argument('--kd', default='pad_euc', help='Hint loss')
args = parser.parse_args()
print(args)


def load_dataset():
    if args.dataset == 'VOC':
        from data import VOCDetection
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(train_sets, args.size)
        epoch_size = len(dataset) // args.batch_size
        max_iter = 70 * epoch_size
        testset = VOCDetection([('2007', 'test')], args.size)
    elif args.dataset == 'COCO':
        from data import COCODetection
        train_sets = [('2017', 'train')]
        dataset = COCODetection(train_sets, args.size)
        epoch_size = len(dataset) // args.batch_size
        max_iter = 140 * epoch_size
        testset = COCODetection([('2017', 'val')], args.size)
    elif args.dataset == 'XML':
        from data import XMLDetection
        dataset = XMLDetection('train', args.size)
        epoch_size = len(dataset) // args.batch_size
        max_iter = 100 * epoch_size
        testset = XMLDetection('val', args.size)
    else:
        raise NotImplementedError('Unkown dataset {}!'.format(args.dataset))
    return (dataset, epoch_size, max_iter, testset)


def save_weights(model, optimizer):
    save_path = os.path.join(args.save_folder, '{}_{}_{}_{}_size{}_anchor{}{}_kd{}_GFocal1011.pth'.format(
        args.dataset,
        ('retina' if args.multi_anchor else 'fcos'),
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


if __name__ == '__main__':

    print('Loading Dataset...')
    (dataset, epoch_size, max_iter, testset) = load_dataset()

    print('Loading student Network...')
    from models.student_detector import Detector
    model = Detector(args.size, dataset.num_classes, args.backbone, args.neck,
        multi_anchor=args.multi_anchor, multi_level=args.multi_level, pretrained=args.pretrained).cuda()
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total param of student model is : {:e}'.format(num_param))

    print('Loading teacher Network...')
    assert args.backbone in ['resnet18', 'repvgg-A0', 'repvgg-A1']
    backbone = 'resnet34' if args.backbone=='resnet18' else 'repvgg-A2'
    neck='pafpn'
    from models.teacher_detector import Detector
    teacher = Detector(args.size, dataset.num_classes, backbone, neck,
        multi_anchor=args.multi_anchor, multi_level=args.multi_level, pretrained=args.pretrained).cuda()
    trained_model = 'weights/public/{}_{}_{}_{}_size{}_anchor{}_MG_GFocal.pth'.format(
            args.dataset, 'retina' if args.multi_anchor else 'fcos', neck, backbone, args.size, args.base_anchor_size)
    print('loading weights from', trained_model)
    state_dict = torch.load(trained_model)
    teacher.load_state_dict(state_dict["model"], strict=True)
    teacher.deploy()
    num_param = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print('Total param is : {:e}'.format(num_param))

    print('Preparing Optimizer & AnchorBoxes...')
    optimizer = optim.SGD(tencent_trick(model), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    ema_model = ModelEMA(model)
    criterion_det = MultiBoxLoss(mutual_guide=args.mutual_guide, multi_anchor=args.multi_anchor)
    criterion_kd = HintLoss(args.kd)
    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size, 
        multi_anchor=args.multi_anchor, multi_level=args.multi_level).cuda()

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
        for iteration in range(max_iter):
            if iteration % epoch_size == 0:

                # random resize
                if iteration < 0.8*max_iter and iteration > args.warm_iter:
                    if args.size == 320:
                        new_size = 64 * (5 + random.choice([-1,0,1]))
                    elif args.size == 512:
                        new_size = 128 * (4 + random.choice([-1,0,1]))
                    else:
                        raise ValueError
                else:
                    new_size = args.size
                print('Switching image size to {}...'.format(new_size))
                dataset.size = new_size
                priors = PriorBox(args.base_anchor_size, new_size, base_size=args.size, 
                    multi_anchor=args.multi_anchor, multi_level=args.multi_level).cuda()

                # create batch iterator
                rand_loader = data.DataLoader(
                    dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=detection_collate
                )
                batch_iterator = iter(rand_loader)
                ema_model.update_attr(model)
                model.train()

            # traning iteratoin
            timer.tic()
            adjust_learning_rate(optimizer, args.lr, iteration, args.warm_iter, max_iter)
            (images, targets) = next(batch_iterator)
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
            with torch.no_grad():
                out_t = teacher(images)
            out = model(images)
            (loss_l, loss_c) = criterion_det(out[:2], priors, targets)
            loss_kd = criterion_kd(out_t[2].detach(), out[2], out_t[1].detach(), out[1], priors, targets)
            loss = loss_l + loss_c + loss_kd
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()
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
        model = ema_model.ema
        save_weights(model, optimizer)
    
    print('Start Evaluation...')
    model.deploy()
    num_images = len(testset)
    all_boxes = [
            [ None for _ in range(num_images) ] for _ in range(testset.num_classes)
        ]
    if args.seq_matcher:
        box_matcher = SeqBoxMatcher()
    if args.draw:
        rgbs = dict()
        os.makedirs("draw/", exist_ok=True)
        os.makedirs("draw/{}/".format(args.dataset), exist_ok=True)
    
    _t = {'im_detect': Timer(), 'im_nms': Timer()}
    for i in range(num_images):
        # prepare image to detect
        img = testset.pull_image(i)
        scale = torch.Tensor(
                [ img.shape[1], img.shape[0], img.shape[1], img.shape[0] ]
            ).cuda()
        x = torch.from_numpy(
                preproc_for_test(img, args.size)
            ).unsqueeze(0).cuda()

        # measure model inference time 
        _t['im_detect'].tic()
        with torch.no_grad():
            out = model(x)
        detect_time = _t['im_detect'].toc()

        # non maximum suppression
        _t['im_nms'].tic()
        (boxes, scores) = Detect(out, priors, scale)
        if args.seq_matcher:
            boxes, scores = box_matcher.update(boxes, scores)
        for j in range(1, testset.num_classes):
            inds = np.where(scores[:, j-1] > args.eval_thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            else:
                all_boxes[j][i] = np.hstack(
                        (boxes[inds], scores[inds, j-1:j])
                    ).astype(np.float32)
        nms_time = _t['im_nms'].toc()

        # draw bounding boxes on images
        if args.draw:
            for j in range(1, testset.num_classes):
                c_dets = all_boxes[j][i]
                for line in c_dets[::-1]:
                    x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                    score = float(line[4])
                    if score > .25:
                        if j not in rgbs:
                            r = random.randint(0,255)
                            g = random.randint(0,255)
                            b = random.randint(0,255)
                            rgbs[j] = [r,g,b]
                        rgb = rgbs[j]
                        label = '{}{:.2f}'.format(testset.pull_classes()[j], score)
                        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
                        cv2.rectangle(img, (x1, y1-15), (x1+len(label)*9, y1), rgb, -1)
                        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            label = 'MutualGuide ({}x{}) : {:.2f}ms on {}'.format(args.size, args.size, detect_time*1000, torch.cuda.get_device_name(0))
            cv2.rectangle(img, (0, 0), (0+len(label)*9, 20), [0,0,0], -1)
            cv2.putText(img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            filename = 'draw/{}/{}.jpg'.format(args.dataset, i)
            cv2.imwrite(filename, img)

        # logging
        if i == 10:
            _t['im_detect'].clear()
            _t['im_nms'].clear()
        if i % math.floor(num_images / 10) == 0 and i > 0:
            print('[{}/{}]Time results: detect={:.2f}ms,nms={:.2f}ms,'.format(
                    i, num_images, detect_time * 1000, nms_time * 1000)
                )
    testset.evaluate_detections(all_boxes)

