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
from data import AnnotationTransform, BaseTransform
from data import detection_collate, preproc
from utils import PriorBox, Detect
from utils import MultiBoxLoss
from utils import Timer
from utils.box import nms
cudnn.benchmark = True
torch.set_printoptions(precision=8)

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
parser.add_argument('--dataset', default='VOC')
parser.add_argument('--save_folder', default='weights/')
parser.add_argument('--mutual_guide', action='store_true')
parser.add_argument('--base_anchor_size', default=24.0, type=float)
parser.add_argument('--size', default=320, type=int)
parser.add_argument('--nms_thresh', default=0.5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--warm_iter', default=500, type=int)
parser.add_argument('--trained_model', help='Location to trained model')
parser.add_argument('--draw', action='store_true', help='Draw detection results')
parser.add_argument('--mixup', action='store_true')
args = parser.parse_args()
print(args)

def load_dataset():
    if args.dataset == 'VOC':
        from data import VOCroot, VOCDetection, VOC_CLASSES
        show_classes = VOC_CLASSES
        num_classes = len(VOC_CLASSES)
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(VOCroot, train_sets, preproc(args.size), AnnotationTransform(), dataset_name='VOC0712trainval')
        epoch_size = len(dataset) // args.batch_size
        max_iter = 250 * epoch_size
        testset = VOCDetection(VOCroot, [('2007', 'test')], None)
    elif args.dataset == 'COCO':
        from data import COCOroot, COCODetection, COCO_CLASSES
        show_classes = COCO_CLASSES
        num_classes = len(COCO_CLASSES)
        train_sets = [('2017', 'train')]
        dataset = COCODetection(COCOroot, train_sets, preproc(args.size))
        epoch_size = len(dataset) // args.batch_size
        max_iter = 140 * epoch_size
        testset = COCODetection(COCOroot, [('2017', 'val')], None)
    elif args.dataset == 'XML':
        from data import XMLroot, XMLDetection, XML_CLASSES
        show_classes = XML_CLASSES
        num_classes = len(XML_CLASSES)
        dataset = XMLDetection(XMLroot, 'train', XML_CLASSES, preproc(args.size))
        epoch_size = len(dataset) // args.batch_size
        max_iter = 140 * epoch_size
        testset = XMLDetection(XMLroot, 'val', XML_CLASSES, None)
    else:
        raise NotImplementedError('Unkown dataset {}!'.format(args.dataset))
    return (show_classes, num_classes, dataset, epoch_size, max_iter, testset)

if __name__ == '__main__':

    print('Loading Dataset...')
    (show_classes, num_classes, dataset, epoch_size, max_iter, testset) = load_dataset()

    print('Loading Network...')
    from models.detector import Detector
    model = Detector(args.size, num_classes, args.backbone, args.neck)
    model.eval()
    model.cuda()
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total param is : {:e}'.format(num_param))
    print('loading weights from', args.trained_model)
    state_dict = torch.load(args.trained_model)
    model.load_state_dict(state_dict, strict=False)

    from models.detector2 import Detector
    backbone = 'resnet34' if args.backbone == 'resnet18' else 'repvgg-A2'
    teacher = Detector(args.size, num_classes, backbone, 'pafpn')
    teacher.eval()
    teacher.cuda()
    num_param = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print('Total param is : {:e}'.format(num_param))
    trained_model = 'weights/{}_pafpn_{}_size320_anchor24.0_MG.pth'.format(args.dataset, backbone)
    print('loading weights from', trained_model)
    state_dict = torch.load(trained_model)
    teacher.load_state_dict(state_dict, strict=True)

    print('Preparing Optimizer & AnchorBoxes...')
    priors = PriorBox(args.base_anchor_size, args.size)
    priors = priors.cuda()


    print('Start Evaluation...')
    transform = BaseTransform(args.size)
    num_images = len(testset)
    conf_diff = list()
    fea_diff = list()
    for i in range(num_images):
        print(i)
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            (x, scale) = (x.cuda(), scale.cuda())

            _, conf1, fea1 = model(x)  # forward pass
            _, conf2, fea2 = teacher(x)  # forward pass
            conf_diff.append((conf1.sigmoid()-conf2.sigmoid()).abs().mean())
            fea_diff.append((fea1-fea2).abs().mean())
    print(sum(conf_diff) / len(conf_diff))
    print(sum(fea_diff) / len(fea_diff))
