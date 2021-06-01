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
from utils import MultiBoxLoss, HintLoss
from utils import Timer
from utils.box import nms
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
parser.add_argument('--kd', default='gauss', help='Hint loss')
args = parser.parse_args()
print(args)


def adjust_learning_rate(optimizer, iteration, warm_iter, max_iter):
    if iteration <= warm_iter:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / warm_iter
    else:
        lr = 1e-6 + (args.lr - 1e-6) * 0.5 * (1 + math.cos((iteration - warm_iter) * math.pi / (max_iter - warm_iter)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def tencent_trick(model):
    (decay, no_decay) = ([], [])
    for (name, param) in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]


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


def save_weights(model):
    save_path = os.path.join(args.save_folder, '{}_{}_{}_size{}_anchor{}{}_kd{}.pth'.format(
        args.dataset,
        args.neck,
        args.backbone,
        args.size,
        args.base_anchor_size,
        ('_MG' if args.mutual_guide else ''),
        args.kd,
        ))
    print('Saving to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    print('Loading Dataset...')
    (show_classes, num_classes, dataset, epoch_size, max_iter, testset) = load_dataset()

    print('Loading student Network...')
    from models.detector_gauss import Detector
    student = Detector(args.size, num_classes, args.backbone, args.neck)
    student.train()
    student.cuda()
    num_param = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print('Total param of student model is : {:e}'.format(num_param))

    print('Loading teacher Network...')
    assert args.backbone in ['resnet18', 'repvgg-A0']
    backbone = 'resnet34' if args.backbone=='resnet18' else 'repvgg-A2'
    from models.detector2 import Detector
    teacher = Detector(args.size, num_classes, backbone, 'pafpn')
    teacher.eval()
    teacher.cuda()
    trained_model = 'weights/{}_pafpn_{}_size320_anchor24.0_MG.pth'.format(args.dataset, 'resnet34' if args.backbone=='resnet18' else 'repvgg-A2')
    print('loading weights from', trained_model)
    state_dict = torch.load(trained_model)
    teacher.load_state_dict(state_dict, strict=True)
    num_param = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print('Total param of teacher model is : {:e}'.format(num_param))

    print('Preparing Optimizer & AnchorBoxes...')
    optimizer = optim.SGD(tencent_trick(student), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    criterion_od = MultiBoxLoss(mutual_guide=args.mutual_guide)
    criterion_kd = HintLoss(mode=args.kd)
    priors = PriorBox(args.base_anchor_size, args.size)
    priors = priors.cuda()

    if args.trained_model is not None:
        print('loading weights from', args.trained_model)
        state_dict = torch.load(args.trained_model)
        student.load_state_dict(state_dict, strict=True)
    else:
        print('Training {}-{} on {} with {} images'.format(args.neck, args.backbone, dataset.name, len(dataset)))
        os.makedirs(args.save_folder, exist_ok=True)
        epoch = 0
        timer = Timer()
        for iteration in range(max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                rand_loader = data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4, collate_fn=detection_collate)
                batch_iterator = iter(rand_loader)
                epoch += 1

            timer.tic()
            adjust_learning_rate(optimizer, iteration, args.warm_iter, max_iter)
            (images, targets) = next(batch_iterator)
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
            
            if args.mixup and iteration < int(0.8*max_iter):
                # mixup
                alpha = 1.0
                lam = np.random.beta(alpha,alpha)
                index = torch.randperm(args.batch_size).cuda()
                inputs = lam*images + (1-lam)*images[index,:]
                targets_a, targets_b = targets, [ targets[index[i]] for i in range(args.batch_size)]

                with torch.no_grad():
                    out_t = teacher(inputs)
                out_s = student(inputs)
                (loss_l_a, loss_c_a) = criterion_od(out_s[:2], priors, targets_a)
                loss_kd_a = criterion_kd(out_t[2].detach(), out_s[2], out_t[1].detach(), out_s[1], priors, targets_a, var=out_s[3])
                (loss_l_b, loss_c_b) = criterion_od(out_s[:2], priors, targets_b)
                loss_kd_b = criterion_kd(out_t[2].detach(), out_s[2], out_t[1].detach(), out_s[1], priors, targets_b, var=out_s[3])
                loss_l = lam * loss_l_a + (1 - lam) * loss_l_b
                loss_c = lam * loss_c_a + (1 - lam) * loss_c_b
                loss_kd = lam * loss_kd_a + (1 - lam) * loss_kd_b
            else:
                # non mixup
                with torch.no_grad():
                    out_t = teacher(images)
                out_s = student(images)
                (loss_l, loss_c) = criterion_od(out_s[:2], priors, targets)
                loss_kd = criterion_kd(out_t[2].detach(), out_s[2], out_t[1].detach(), out_s[1], priors, targets, var=out_s[3])
            loss = loss_l + loss_c + loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            load_time = timer.toc()

            if iteration % 100 == 0:
                print('Epoch {}, iter {}, lr {:.6f}, loss_l {:.2f}, loss_c {:.2f}, loss_kd {:.2f}, loss {:.2f}, time {:.2f}s, eta {:.2f}h'.format(
                    epoch,
                    iteration,
                    optimizer.param_groups[0]['lr'],
                    loss_l.item(),
                    loss_c.item(),
                    loss_kd.item(),
                    loss.item(),
                    load_time,
                    load_time * (max_iter - iteration) / 3600,
                    ))
                timer.clear()
        save_weights(student)
    
    print('Start Evaluation...')
    thresh=0.01
    max_per_image=300
    student.eval()
    for module in student.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    detector = Detect(num_classes)
    transform = BaseTransform(args.size)
    num_images = len(testset)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    rgbs = dict()
    os.makedirs("draw/", exist_ok=True)
    os.makedirs("draw/{}/".format(args.dataset), exist_ok=True)
    _t = {'im_detect': Timer(), 'im_nms': Timer()}
    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            (x, scale) = (x.cuda(), scale.cuda())

            _t['im_detect'].tic()
            out = student(x)  # forward pass
            (boxes, scores) = detector.forward(out[:2], priors)
            detect_time = _t['im_detect'].toc()

        boxes *= scale  # scale each detection back up to the image
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        _t['im_nms'].tic()
        for j in range(1, num_classes):
            inds = np.where(scores[:, j - 1] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j - 1]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(c_dets, thresh=args.nms_thresh)  # non maximum suppression
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['im_nms'].toc()

        if args.draw:
            for j in range(1, num_classes):
                c_dets = all_boxes[j][i]
                for line in c_dets:
                    x1 = int(line[0])
                    y1 = int(line[1])
                    x2 = int(line[2])
                    y2 = int(line[3])
                    score = float(line[4])
                    if score > .25:
                        if j not in rgbs:
                            r = random.randint(0,255)
                            g = random.randint(0,255)
                            b = random.randint(0,255)
                            rgbs[j] = [r,g,b]
                        rgb = rgbs[j]
                        label = '{}{:.2f}'.format(show_classes[j], score)
                        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
                        cv2.rectangle(img, (x1, y1-15), (x1+len(label)*9, y1), rgb, -1)
                        img = cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'Resolution {}x{} detect {:.2f}ms on {}'.format(args.size, args.size, detect_time*1000, torch.cuda.get_device_name(0)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            filename = 'draw/{}/{}.jpg'.format(args.dataset, i)
            cv2.imwrite(filename, img)

        if i == 10:
            _t['im_detect'].clear()
            _t['im_nms'].clear()
        if i % math.floor(num_images / 10) == 0 and i > 0:
            print('[{}/{}]Time results: detect={:.2f}ms,nms={:.2f}ms,'.format(i, num_images, detect_time * 1000, nms_time * 1000))
    testset.evaluate_detections(all_boxes)

