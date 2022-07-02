#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import os.path
import torch.utils.data as data
import cv2
import numpy as np
from .voc_eval import voc_eval
from .data_augment import preproc_for_train
import xml.etree.ElementTree as ET


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox"""

    def __init__(self, keep_difficult=True):
        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):

        width = float(target.find("size").find("width").text)
        height = float(target.find("size").find("height").text)

        res = np.empty((0, 5))
        for obj in target.iter("object"):

            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue

            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)
            for i, pt in enumerate([width, height, width, height]):
                bndbox[i] /= pt

            name = obj.find("name").text.lower().strip()
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object"""

    def __init__(
        self,
        image_sets: list,
        size: int = 320,
        dataset_name: str = "VOC0712",
        cache: bool = False,
    ) -> None:
        self.root = os.path.join("/srv/tempdd/xwang/voc/", "VOCdevkit/")
        self.image_set = image_sets
        self.size = size
        self.target_transform = AnnotationTransform()
        self.name = dataset_name + str(image_sets)
        self.num_classes = len(self.pull_classes())
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            for line in open(
                os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            ):
                self.ids.append((rootpath, line.strip()))

    def pull_classes(
        self,
    ) -> tuple:

        return VOC_CLASSES

    def __getitem__(
        self,
        index,
    ) -> list:
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
    ) -> np.ndarray:
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        return self.target_transform(target)

    def pull_image(
        self,
        index: int,
        resize: bool = False,
    ) -> np.ndarray:
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        if resize:
            image = cv2.resize(
                image, (self.size, self.size), interpolation=cv2.INTER_LINEAR
            )
        return image

    def evaluate_detections(
        self,
        all_boxes: list,
    ) -> float:
        output_dir = os.path.join(self.root, "eval")
        os.makedirs(output_dir, exist_ok=True)
        self._write_voc_results_file(all_boxes)
        results = []
        for thresh in np.arange(0.5, 1, 0.05):
            result = self._do_python_eval(output_dir, thresh)
            results.append(result)
            print("----thresh={:.2f}, AP={:.3f}".format(thresh, result))

        print(
            "mAP results: AP50={:.3f}, AP75={:.3f}, AP={:.3f}".format(
                results[0], results[5], sum(results) / 10
            )
        )
        return sum(results) / 10

    def _get_voc_results_file_template(
        self,
    ) -> str:
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(
        self,
        all_boxes: list,
    ) -> None:
        for cls_ind, cls in enumerate(VOC_CLASSES):
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(
        self,
        output_dir: str = "output",
        thresh: float = 0.5,
    ) -> float:
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(self.root, "annotations_cache")
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for cls in VOC_CLASSES:
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=thresh,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            # print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        return np.mean(aps)
