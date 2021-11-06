#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import math
import torch


def _crop(image, boxes, labels, p=0.75, min_iou=0.75, max_iou=0.25):

    def matrix_iou(a, b):
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / area_a[:, np.newaxis]

    if random.random() > p:
        return (image, boxes, labels)

    (height, width, _) = image.shape
    while True:
        scale = random.uniform(0.5, 1.)
        min_ratio = max(0.5, scale * scale)
        max_ratio = min(2, 1. / scale / scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        w = int(scale * ratio * width)
        h = int(scale / ratio * height)

        l = random.randrange(width - w)
        t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        iou = matrix_iou(boxes, roi[np.newaxis])
        iou = iou[iou < min_iou]
        iou = iou[iou >= max_iou]
        if len(iou) > 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask].copy()
        labels_t = labels[mask].copy()
        if len(boxes_t) == 0:
            continue

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        return (image_t, boxes_t, labels_t)


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, labels, p=0.75):

    if random.random() > p:
        return (image, boxes, labels)

    (height, width, depth) = image.shape
    while True:
        scale = random.uniform(1, 2)
        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale * scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.ones((h, w, depth)) * 114.0
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return (image, boxes_t, labels)


def _mirror(image, boxes):
    (_, width, _) = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return (image, boxes)


def preproc_for_test(image, insize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap=(2, 0, 1)):
    image = cv2.resize(image, (insize, insize), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std
    image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image


def preproc_for_train(image, targets, insize):
    boxes = targets[:, :-1].copy()
    labels = targets[:, -1].copy()
    if len(boxes) == 0:
        targets = np.zeros((1, 5))
        image = preproc_for_test(image, insize)
        return (torch.from_numpy(image), targets)

    image_o = image.copy()
    targets_o = targets.copy()
    (height_o, width_o, _) = image_o.shape
    boxes_o = targets_o[:, :-1]
    labels_o = targets_o[:, -1]
    boxes_o[:, 0::2] /= width_o
    boxes_o[:, 1::2] /= height_o
    labels_o = np.expand_dims(labels_o, 1)
    targets_o = np.hstack((boxes_o, labels_o))

    image_t = _distort(image)
    (image_t, boxes, labels) = _crop(image_t, boxes, labels)
    (image_t, boxes, labels) = _expand(image_t, boxes, labels)
    (image_t, boxes) = _mirror(image_t, boxes)

    (height, width, _) = image_t.shape
    image_t = preproc_for_test(image_t, insize)
    boxes = boxes.copy()
    boxes[:, 0::2] /= width
    boxes[:, 1::2] /= height
    b_w = (boxes[:, 2] - boxes[:, 0]) * 1.
    b_h = (boxes[:, 3] - boxes[:, 1]) * 1.
    mask_b = np.minimum(b_w, b_h) > (8. / insize)
    boxes_t = boxes[mask_b]
    labels_t = labels[mask_b].copy()

    if len(boxes_t) == 0:
        image = preproc_for_test(image_o, insize)
        return (torch.from_numpy(image), targets_o)

    labels_t = np.expand_dims(labels_t, 1)
    targets_t = np.hstack((boxes_t, labels_t))
    return (torch.from_numpy(image_t), targets_t)


def detection_collate(batch):
    """ Custom collate fn for images and boxes """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                annos.requires_grad = False
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
