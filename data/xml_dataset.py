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
from .voc_eval import voc_eval
import pickle
import numpy as np
from .data_augment import preproc_for_train

XMLroot = 'datasets/XML/'
XML_CLASSES = ( '__background__', # always index 0
    'person','bicycle','car','motorbike','aeroplane',
    'bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant',
    'bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli',
    'carrot','hot dog','pizza','donut','cake','chair',
    'sofa','pottedplant','bed','diningtable','toilet',
    'tvmonitor','laptop','mouse','remote','keyboard',
    'cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush')

class AnnotationTransform(object):
    def __init__(self, classes):
        self.class_to_ind = dict(zip(classes, range(len(classes))))
    def __call__(self, target):
        res = np.empty((0,5)) 
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))
        return res

class XMLDetection(data.Dataset):
    def __init__(self, image_sets, size):
        self.root = XMLroot
        self.image_set = image_sets
        self.size = size
        self.target_transform = AnnotationTransform(XML_CLASSES)
        self._annopath = os.path.join(self.root, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, 'JPEGImages', '%s.jpg')
        self.num_classes = len(self.pull_classes())
        self.ids = list()
        self.name = os.path.join(self.root, self.image_set + '.txt')
        for line in open(self.name):
            self.ids.append(line.strip())
        print('Using custom dataset. Reading {}...'.format(self.name))

    def pull_classes(self):
        return XML_CLASSES

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        target = self.target_transform(target)
        img, target = preproc_for_train(img, target, self.size)
        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def evaluate_detections(self, all_boxes):
        output_dir = os.path.join(self.root, 'eval')
        self._write_voc_results_file(all_boxes)
        results = []
        for thresh in np.arange(0.5,1,0.05):
            result = self._do_python_eval(output_dir, thresh)
            results.append(result)
            print('----thresh={:.2f}, AP={:.3f}'.format(thresh, result))

        print('mAP results: AP50={:.3f}, AP75={:.3f}, AP={:.3f}'.format(results[0], results[5], sum(results)/10))
        return results

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(self.root, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(XML_CLASSES):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output', thresh=0.5):
        rootpath = self.root
        name = self.image_set
        annopath = os.path.join(rootpath, 'Annotations', '{:s}_PreviewData.xml')
        imagesetfile = os.path.join(rootpath, name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []
        use_07_metric = True
        #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(XML_CLASSES):

            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=thresh,
                                    use_07_metric=use_07_metric)
            aps += [ap]
            if thresh == 0.5:
                print('AP50 for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        #print('Thresh = {:.4f} Mean AP = {:.4f}'.format(thresh, np.mean(aps)))
        """print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')"""
        return np.mean(aps)
