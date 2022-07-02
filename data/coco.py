#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch.utils.data as data
import cv2
import numpy as np
import json
import tempfile
from .data_augment import preproc_for_train

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODetection(data.Dataset):
    """COCO Detection Dataset Object"""

    def __init__(
        self,
        image_sets: list,
        size: int,
        dataset_name: str = "COCO2017",
    ) -> None:
        self.root = os.path.join("/home/heng/Documents/Datasets/", "COCO/")
        self.size = size
        self.name = dataset_name + str(image_sets) + str(self.size)
        self.ids = list()
        self.annotations = list()
        for (year, image_set) in image_sets:
            self.coco_name = image_set + year
            prefix = "instances" if self.coco_name.find("test") == -1 else "image_info"
            annofile = os.path.join(
                self.root, "annotations", prefix + "_" + self.coco_name + ".json"
            )
            self._COCO = COCO(annofile)
            cats = self._COCO.loadCats(self._COCO.getCatIds())
            self._classes = tuple([c["name"] for c in cats])
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
            self._class_to_coco_cat_id = dict(
                zip([c["name"] for c in cats], self._COCO.getCatIds())
            )
            self.image_indexes = self._COCO.getImgIds()
            self.ids.extend(
                [
                    self.image_path_from_index(self.coco_name, index)
                    for index in self.image_indexes
                ]
            )
            # test set will not load annotations
            if image_set.find("test") == -1:
                self.annotations.extend(
                    [
                        self._annotation_from_index(index, self._COCO)
                        for index in self.image_indexes
                    ]
                )
                # val set will not remove non-valid images
                if image_set.find("val") == -1:
                    ids, annotations = [], []
                    for i, a in zip(self.ids, self.annotations):
                        if a.shape[0] > 0:
                            ids.append(i)
                            annotations.append(a)
                    self.ids = ids
                    self.annotations = annotations

    def pull_classes(
        self,
    ) -> list:
        return self._classes

    def image_path_from_index(
        self,
        name: str,
        index: int,
    ) -> str:
        """Construct an image path"""
        file_name = str(index).zfill(12) + ".jpg"
        return os.path.join(self.root, name, file_name)

    def _annotation_from_index(
        self,
        index: int,
        _COCO,
    ) -> np.ndarray:
        """Loads COCO bounding-box instance annotations"""
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and (x2 - x1) > 6 and (y2 - y1) > 6:
                obj["clean_bbox"] = [x1 / width, y1 / height, x2 / width, y2 / height]
                valid_objs.append(obj)
        num_objs = len(valid_objs)
        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class indices
        coco_cat_id_to_class_ind = dict(
            [
                (self._class_to_coco_cat_id[cls], self._class_to_ind[cls])
                for cls in self._classes
            ]
        )

        for ix, obj in enumerate(valid_objs):
            cls = coco_cat_id_to_class_ind[obj["category_id"]]
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        return res

    def pull_image(
        self,
        index: int,
        resize: bool = False,
    ) -> np.ndarray:
        """Returns the original image object at index"""
        img_id = self.ids[index]
        image = cv2.imread(img_id, cv2.IMREAD_COLOR)
        if resize:
            image = cv2.resize(
                image, (self.size, self.size), interpolation=cv2.INTER_LINEAR
            )
        return image

    def __getitem__(
        self,
        index: int,
    ) -> list:
        target = self.annotations[index]
        img = self.pull_image(index, resize=True)
        img, target = preproc_for_train(img, target, self.size)
        return img, target

    def __len__(
        self,
    ) -> int:
        return len(self.ids)

    def _coco_results_one_category(
        self,
        boxes: np.ndarray,
        cat_id: int,
    ) -> list:
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
                [
                    {
                        "image_id": index,
                        "category_id": cat_id,
                        "bbox": [xs[k], ys[k], ws[k], hs[k]],
                        "score": scores[k],
                    }
                    for k in range(dets.shape[0])
                ]
            )
        return results

    def _coco_detection_metrics(
        self,
        res_file: str,
    ) -> float:
        def _get_thr_ind(coco_eval, thr):
            ind = np.where(
                (coco_eval.params.iouThrs > thr - 1e-5)
                & (coco_eval.params.iouThrs < thr + 1e-5)
            )[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        coco_eval = COCOeval(self._COCO, self._COCO.loadRes(res_file))
        coco_eval.params.useSegm = False
        coco_eval.evaluate()
        coco_eval.accumulate()

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)

        precision = coco_eval.eval["precision"][ind_lo : (ind_hi + 1), :, :, 0, 2]
        coco_eval.summarize()
        return np.mean(precision[precision > -1])

    def evaluate_detections(
        self,
        all_boxes: np.ndarray,
    ) -> float:
        results = []
        for cls_ind, cls in enumerate(self._classes):
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(
                self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id)
            )
        _, res_file = tempfile.mkstemp()
        with open(res_file, "w") as fid:
            json.dump(results, fid)
        # Only do evaluation on non-test sets
        if self.coco_name.find("test") == -1:
            return self._coco_detection_metrics(res_file)
        else:
            return float("nan")
