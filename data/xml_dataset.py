import os
import os.path
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from .voc_eval import voc_eval
import pickle
import numpy as np

XMLroot = 'datasets/FLIR/'
XML_CLASSES = ['__background__', 'person', 'car', 'bicycle', 'dog']

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
    def __init__(self, root, image_sets, classes, preproc=None):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.classes = classes
        self.target_transform = AnnotationTransform(self.classes)
        self._annopath = os.path.join(self.root, 'Annotations', '%s.xml')
        self.ids = list()
        self.name = os.path.join(self.root, self.image_set + '.txt')
        for line in open(self.name):
            self.ids.append(line.strip())
        print('Using custom dataset. Reading {}...'.format(self.name))

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = self.pull_image(index)
        target = self.target_transform(target)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_id(self, index):
        return self.ids[index]

    def pull_classes(self):
        return KAIST_CLASSES

    def pull_image(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, 'JPEGImages', '{}.jpeg'.format(img_id))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img

    def pull_anno(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        return self.target_transform(target)

    def evaluate_detections(self, all_boxes):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
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
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            #print('Writing {} VOC results file'.format(cls))
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
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.classes):

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