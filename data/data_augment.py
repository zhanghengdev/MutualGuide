#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import math
import torch
from PIL import Image

def _gridmask(
    image: np.ndarray, 
    boxes: np.ndarray, 
    ratio: float = 0.5, 
    rotate: int = 1, 
    img_mean: float = 114.0, 
    p: float = 0.7, 
) -> np.ndarray:

    if random.random() > p:
        return (image, boxes)

    (height, width, depth) = image.shape
    hh = int(1.5*height)
    ww = int(1.5*width)
    d1 = 2
    d2 = max(np.min(boxes[:,2:]-boxes[:,:2]) * min(height, width), d1+1)
    d = np.random.randint(d1, d2)
    l = min(max(int(d*ratio+0.5),1),d-1)
    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(hh//d):
        s = d*i + st_h
        t = min(s+l, hh)
        mask[s:t,:] *= 0
    for i in range(ww//d):
        s = d*i + st_w
        t = min(s+l, ww)
        mask[:,s:t] *= 0
   
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh-height)//2:(hh-height)//2+height, (ww-width)//2:(ww-width)//2+width]
    mask = np.dstack((mask, mask, mask)) 
    image = image * (1. - mask) + img_mean * mask

    return (image, boxes)

def _crop_expand(
    image: np.ndarray, 
    boxes: np.ndarray, 
    labels: np.ndarray, 
    min_scale: float = 0.25, 
    max_scale: float = 1.75, 
    min_ratio: float = 0.5, 
    max_ratio: float = 1.0, 
    min_shift: float = 0.4, 
    max_shift: float = 0.6, 
    min_iou: float = 0.75, 
    max_iou: float = 0.25, 
    max_try: int = 10, 
    img_mean: float = 114.0, 
    p: float = 0.75, 
) -> np.ndarray:

    def matrix_iou(a, b):
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / area_a[:, np.newaxis]

    if random.random() > p:
        return (image, boxes, labels)

    (height, width, depth) = image.shape
    assert height == width
    
    for _ in range(max_try):
        new_h = random.uniform(min_scale, max_scale)
        if random.randrange(2):
            new_w = new_h * random.uniform(min_ratio, max_ratio)
        else:
            new_w = new_h / random.uniform(min_ratio, max_ratio)

        for _ in range(max_try):
            center_y = random.uniform(min_shift, max_shift)
            center_x = random.uniform(min_shift, max_shift)
            corner_y = center_y - new_h/2
            corner_x = center_x - new_w/2

            cropped_y1 = max(0, corner_y)
            cropped_x1 = max(0, corner_x)
            cropped_y2 = min(1.0, corner_y+new_h)
            cropped_x2 = min(1.0, corner_x+new_w)
            expand_y1 = max(0, -corner_y)
            expand_x1 = max(0, -corner_x)
            
            real_cropped_y1 = int(cropped_y1 * height)
            real_cropped_x1 = int(cropped_x1 * width)
            real_cropped_y2 = int(cropped_y2 * height)
            real_cropped_x2 = int(cropped_x2 * width)
            real_expand_y1 = int(expand_y1 * height)
            real_expand_x1 = int(expand_x1 * width)
            real_expand_y2 = real_expand_y1 + real_cropped_y2 - real_cropped_y1
            real_expand_x2 = real_expand_x1 + real_cropped_x2 - real_cropped_x1

            cropped_image = image[
                real_cropped_y1 : real_cropped_y2, real_cropped_x1 : real_cropped_x2
            ]
            expand_image = np.ones(
                (math.ceil(height * new_h), math.ceil(width * new_w), depth)
            ) * img_mean
            expand_image[
                real_expand_y1:real_expand_y2, real_expand_x1:real_expand_x2
            ] = cropped_image

            roi = np.array((cropped_x1, cropped_y1, cropped_x2, cropped_y2))
            iou = matrix_iou(boxes, roi[np.newaxis])
            iou = iou[iou < min_iou]
            iou = iou[iou > max_iou]
            if len(iou) > 0:
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, 0] = np.maximum(0, boxes_t[:, 0]-corner_x) / new_w
            boxes_t[:, 1] = np.maximum(0, boxes_t[:, 1]-corner_y) / new_h
            boxes_t[:, 2] = np.minimum(new_w, boxes_t[:, 2]-corner_x) / new_w
            boxes_t[:, 3] = np.minimum(new_h, boxes_t[:, 3]-corner_y) / new_h

            return (expand_image, boxes_t, labels_t)

    return (image, boxes, labels)


def _distort(
    image: np.ndarray, 
) -> np.ndarray:

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


def _mirror(
    image: np.ndarray, 
    boxes: np.ndarray, 
) -> tuple:

    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = 1.0 - boxes[:, 2::-2]
    return (image, boxes)


def preproc_for_test(
    image: np.ndarray, 
    insize: int, 
    mean: list = (0.485, 0.456, 0.406),
    std: list = (0.229, 0.224, 0.225),
    swap: list = (2, 0, 1),
) -> tuple:

    image = cv2.resize(image, (insize, insize), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    image -= mean
    image /= std
    image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image


def preproc_for_train(
    image: np.ndarray,
    targets: np.ndarray,
    insize: int,
) -> tuple:

    assert len(targets) != 0,  'ERROR: No objects found for augmentation'

    boxes = targets[:, :-1].copy()
    labels = targets[:, -1].copy()

    image = _distort(image)
    (image, boxes) = _gridmask(image, boxes)
    (image, boxes, labels) = _crop_expand(image, boxes, labels)
    (image, boxes) = _mirror(image, boxes)

    image = preproc_for_test(image, insize)
    labels = np.expand_dims(labels, 1)
    targets = np.hstack((boxes, labels))

    return (torch.from_numpy(image), targets)


def detection_collate(
    batch: tuple,
) -> tuple:

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
