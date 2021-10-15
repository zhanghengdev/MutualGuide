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
from data import detection_collate, preproc_for_test
from utils import PriorBox, Detect
from utils import MultiBoxLoss
from utils import Timer, ModelEMA
from utils import adjust_learning_rate, tencent_trick
from utils.box import SeqBoxMatcher
cudnn.benchmark = True
import time

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
parser.add_argument('--multi_anchor', action='store_true')
parser.add_argument('--multi_level', action='store_true')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--seq_matcher', action='store_true')
parser.add_argument('--base_anchor_size', default=24.0, type=float)
parser.add_argument('--size', default=320, type=int)
parser.add_argument('--eval_thresh', default=0.01, type=float)
parser.add_argument('--nms_thresh', default=0.5, type=float)
parser.add_argument('--trained_model', help='Location to trained model')
parser.add_argument('--draw', action='store_true', help='Draw detection results')
args = parser.parse_args()
print(args)


if __name__ == '__main__':

    print('Loading Dataset...')
    if args.dataset == 'VOC':
        from data import VOCDetection
        testset = VOCDetection([('2007', 'test')], args.size)
    elif args.dataset == 'COCO':
        from data import COCODetection
        testset = COCODetection([('2017', 'val')], args.size)
    elif args.dataset == 'XML':
        from data import XMLDetection
        testset = XMLDetection('val', args.size)
    else:
        raise NotImplementedError('Unkown dataset {}!'.format(args.dataset))

    print('Loading Network...')
    from models.teacher_detector import Detector
    model = Detector(args.size, testset.num_classes, args.backbone, args.neck,
        multi_anchor=args.multi_anchor, multi_level=args.multi_level, pretrained=args.pretrained).cuda()

    print('Preparing AnchorBoxes...')
    priors = PriorBox(args.base_anchor_size, args.size, base_size=args.size, 
        multi_anchor=args.multi_anchor, multi_level=args.multi_level).cuda()

    print('Loading weights from', args.trained_model)
    state_dict = torch.load(args.trained_model)
    keys = list(state_dict["model"].keys())
    for k in keys:
        if 'dist' in k:
            state_dict["model"].pop(k)
    model.load_state_dict(state_dict["model"], strict=True)
    model.deploy()
    
    print('Start Evaluation...')
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
        start = time.time()
        with torch.no_grad():
            out = model.forward_test(x)
        runtime = (time.time()-start)*1000

        # non maximum suppression
        start = time.time()
        (boxes, scores) = Detect(out, priors, scale, 
            eval_thresh=args.eval_thresh, nms_thresh=args.nms_thresh)
        nmstime = (time.time()-start)*1000

        # logging
        print('[{}/{}] runtime={:.2f}ms nms={:.2f}ms        '.format(
            i, num_images, runtime, nmstime), end="\r")

        # post processing
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

        # draw bounding boxes on images
        if args.draw:
            for j in range(1, testset.num_classes):
                c_dets = all_boxes[j][i]
                for line in c_dets[::-1]:
                    x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                    score = float(line[4])
                    if score > .5:
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

    testset.evaluate_detections(all_boxes)
