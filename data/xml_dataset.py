#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path
import random
import torch
import torch.utils.data as data
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import numpy as np
from .data_augment import preproc_for_train

XML_CLASSES = ('__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class XMLDetection(data.Dataset):
    def __init__(
        self,
        image_sets: list,
        size: float = 320, 
    ) -> None:
        self.root = os.path.join('datasets/', 'XML/')
        self.image_set = image_sets
        self.size = size
        self.classes = XML_CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self._annopath = os.path.join(self.root, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, 'JPEGImages', '%s.jpg')
        self.num_classes = len(self.pull_classes())
        self.ids = list()
        self.name = 'XML-'+str(self.image_set)
        self.file_name = os.path.join(self.root, self.image_set + '.txt')
        for line in open(self.file_name):
            self.ids.append(line.strip())
        print('Using custom dataset. Reading {}...'.format(self.file_name))

    def pull_classes(
        self,
    ) -> tuple:
        return self.classes

    def __getitem__(
        self,
        index: int,
    ) -> list:
        img_id = self.ids[index]
        img = self.pull_image(index, resize=True)
        target = self.pull_anno(index)
        img, target = preproc_for_train(img, target, self.size)
        return img, target

    def __len__(
        self,
    ) -> int:
        return len(self.ids)

    def pull_anno(
        self,
        index: int,
        normalize: bool = True,
    ) -> np.ndarray:
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        res = np.empty((0,5)) 
        width = float(target.find('size').find('width').text)
        height = float(target.find('size').find('height').text)
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            if normalize:
                for i, pt in enumerate([width, height, width, height]):
                    bndbox[i] /= pt
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))
        return res

    def pull_image(
        self,
        index: int,
        resize: bool = False,
    ) -> np.ndarray:
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        if resize:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image

    def evaluate_detections(
        self,
        all_boxes: list,
    ) -> float:
        results = []
        for thresh in np.arange(0.5,1,0.05):
            result = self.calculate_map(all_boxes, thresh)
            results.append(result)
            print('----thresh={:.2f}, AP={:.3f}'.format(thresh, result))

        print('mAP results: AP50={:.3f}, AP75={:.3f}, AP={:.3f}'.format(results[0], results[5], sum(results)/10))
        return sum(results)/10

    def calculate_map(
        self,
        all_boxes: list,
        thresh: float,
    ) -> float:
        
        aps = list()
        for j in range(1, self.num_classes):

            # prepare gt
            class_recs = list()
            npos = 0
            for i in range(len(self)):
                R = dict()
                anno = self.pull_anno(i, normalize=False)
                inds = np.where(anno[:, -1] == j)[0]
                if len(inds) == 0:
                    R['bbox'] = np.empty([0, 4], dtype=np.float32)
                else:
                    R['bbox'] = anno[inds, :4]
                R['det'] = [False] * len(inds)
                class_recs.append(R)
                npos += len(inds)

            # parse det
            image_ids = list()
            confidence = list()
            BB = np.empty([0, 4], dtype=np.float32)
            for i in range(len(self)):
                for det in all_boxes[j][i]:
                    image_ids.append(i)
                    confidence.append(det[-1])
                    BB = np.vstack((BB,det[np.newaxis, :4]))
            image_ids = np.array(image_ids)
            confidence = np.array(confidence)

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            image_ids = image_ids[sorted_ind]
            sorted_scores = confidence[sorted_ind]
            BB = BB[sorted_ind, :]

            # mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[int(image_ids[d])]
                BBGT = R['bbox'].astype(float)
                bb = BB[d, :].astype(float)
                ovmax = -np.inf

                if BBGT.size > 0:
                    # compute overlaps
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                    uni = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                          (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - \
                          inters
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if (ovmax > thresh) and (not R['det'][jmax]):
                    R['det'][jmax] = True
                    tp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            aps.append(ap)

        return np.mean(aps)