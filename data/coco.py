#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
from .data_augment import preproc_for_train

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODetection(data.Dataset):
    """ COCO Detection Dataset Object """

    def __init__(self, image_sets, size, dataset_name='COCO2017', cache=True):
        self.root = os.path.join('datasets/', 'coco2017/')
        self.cache_path = os.path.join(self.root, 'cache')
        self.image_set = image_sets
        self.size = size
        self.name = dataset_name+str(self.image_set)+str(self.size)
        self.ids = list()
        self.annotations = list()
        for (year, image_set) in image_sets:
            coco_name = image_set+year
            annofile = self._get_ann_file(coco_name)
            self._COCO = COCO(annofile)
            self.coco_name = coco_name
            cats = self._COCO.loadCats(self._COCO.getCatIds())
            self._classes = tuple(['__background__'] + [c['name'] for c in cats])
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
            self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats], self._COCO.getCatIds()))
            indexes = self._COCO.getImgIds()
            self.image_indexes = indexes
            self.ids.extend([self.image_path_from_index(coco_name, index) for index in indexes])
            if image_set.find('test') != -1:
                print('test set will not load annotations!')
            else:
                self.annotations.extend(self._load_coco_annotations(indexes, self._COCO))
                if image_set.find('val') != -1:
                    print('val set will not remove non-valid images!')
                else:
                    ids, annotations = [], []
                    for i, a in zip(self.ids, self.annotations):
                        if a.shape[0] > 0:
                            ids.append(i)
                            annotations.append(a)
                    self.ids = ids
                    self.annotations = annotations
        if cache:
            self._cache_images()
        else:
            self.imgs = None

    def pull_classes(self):
        return self._classes

    def image_path_from_index(self, name, index):
        """ Construct an image path """
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = os.path.join(self.root, name, file_name)
        # assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_ann_file(self, name):
        prefix = 'instances' if name.find('test') == -1 else 'image_info'
        return os.path.join(self.root, 'annotations', prefix + '_' + name + '.json')

    def _load_coco_annotations(self, indexes, _COCO):
        gt_roidb = [self._annotation_from_index(index, _COCO) for index in indexes]
        return gt_roidb

    def _annotation_from_index(self, index, _COCO):
        """ Loads COCO bounding-box instance annotations """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and (x2-x1) > 6 and (y2-y1) > 6:
                obj['clean_bbox'] = [x1/width, y1/height, x2/width, y2/height]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        return res

    def _cache_images(self):
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            print(
                "Caching images for the first time..."
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), self.size, self.size, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.pull_image(x, resize=True),
                range(len(self.ids)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.ids))
            for k, out in pbar:
                self.imgs[k] = out.copy()
            self.imgs.flush()
            pbar.close()
        
        print("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), self.size, self.size, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def pull_image(self, index, resize=False):
        ''' Returns the original image object at index '''
        img_id = self.ids[index]
        image = cv2.imread(img_id, cv2.IMREAD_COLOR)
        if resize:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.annotations[index]
        if self.imgs is not None:
            img = self.imgs[index].copy()
        else:
            img = self.pull_image(index, resize=True)
        img, target = preproc_for_train(img, target, self.size)
        return img, target

    def __len__(self):
        return len(self.ids)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('all: {:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            #print('{}: {:.1f}'.format(cls, 100 * ap))
            #print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_indexes):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id' : index,
                'category_id' : cat_id,
                'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                'score' : scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            #print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes ))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes):
        output_dir = os.path.join(self.root, 'eval')
        os.makedirs(output_dir, exist_ok=True)
        res_file = os.path.join(output_dir, ('detections_' + self.coco_name + '_results'))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file

