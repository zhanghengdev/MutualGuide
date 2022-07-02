#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import argparse
import math
import numpy as np
import cv2
import random
import torch
import torch.backends.cudnn as cudnn
from data import preproc_for_test
from models.detector import Detector
from utils import Timer, PriorBox, Detect, SeqBoxMatcher

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

parser = argparse.ArgumentParser(description="Model Evluation")
parser.add_argument("--dataset", default="COCO")
parser.add_argument("--neck", default="pafpn")
parser.add_argument("--backbone", default="resnet18")
parser.add_argument("--anchor_size", default=48.0, type=float)
parser.add_argument("--image_size", default=640, type=int)
parser.add_argument("--eval_thresh", default=0.05, type=float)
parser.add_argument("--nms_thresh", default=0.5, type=float)
parser.add_argument("--trained_model", help="Location to trained model")
parser.add_argument("--vis", action="store_true", help="vis detection results")
parser.add_argument("--seq_matcher", action="store_true", help="Video Object Detection")
args = parser.parse_args()
print(args)


if __name__ == "__main__":

    print("Loading Dataset...")
    if args.dataset == "COCO":
        from data import COCODetection

        testset = COCODetection([("2017", "val")], args.image_size)
    elif args.dataset == "VOC":
        from data import VOCDetection

        testset = VOCDetection([("2007", "test")], args.image_size)
    elif args.dataset == "XML":
        from data import XMLDetection

        testset = XMLDetection("val", args.image_size)
    else:
        raise NotImplementedError("Unkown dataset {}!".format(args.dataset))

    print("Loading Network...")
    model = Detector(
        args.image_size, testset.num_classes, args.backbone, args.neck, mode="normal"
    ).cuda()

    print("Loading weights from", args.trained_model)
    state_dict = torch.load(args.trained_model)
    model.load_state_dict(state_dict["model"], strict=False)
    model.deploy()

    print("Evaluating model complexity...")
    from utils import get_model_complexity_info

    flops, params = get_model_complexity_info(
        model, (3, args.image_size, args.image_size)
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", flops))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))

    print("Preparing AnchorBoxes...")
    priors = PriorBox(args.anchor_size, args.image_size).cuda()

    print("Start Evaluation...")
    num_images = len(testset)
    all_boxes = [[None for _ in range(num_images)] for _ in range(testset.num_classes)]
    if args.seq_matcher:
        box_matcher = SeqBoxMatcher()
    if args.vis:
        rgbs = dict()
        os.makedirs("vis/", exist_ok=True)
        os.makedirs("vis/{}/".format(args.dataset), exist_ok=True)
    _t = {"im_detect": Timer(), "im_nms": Timer()}
    for i in range(num_images):

        # prepare image to detect
        img = testset.pull_image(i)
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        ).cuda()
        x = torch.from_numpy(preproc_for_test(img, args.image_size)).unsqueeze(0).cuda()

        # model inference
        torch.cuda.current_stream().synchronize()
        _t["im_detect"].tic()
        with torch.no_grad():
            out = model(x)
        torch.cuda.current_stream().synchronize()
        detect_time = _t["im_detect"].toc()

        # post processing
        _t["im_nms"].tic()
        (boxes, scores) = Detect(
            out,
            priors,
            scale,
            eval_thresh=args.eval_thresh,
            nms_thresh=args.nms_thresh,
        )
        if args.seq_matcher:
            boxes, scores = box_matcher.update(boxes, scores)
        for j in range(testset.num_classes):
            inds = np.where(scores[:, j] > args.eval_thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            else:
                all_boxes[j][i] = np.hstack(
                    (boxes[inds], scores[inds, j : j + 1])
                ).astype(np.float32)
        nms_time = _t["im_nms"].toc()

        # vis bounding boxes on images
        if args.vis:
            for j in range(testset.num_classes):
                c_dets = all_boxes[j][i]
                for line in c_dets[::-1]:
                    x1, y1, x2, y2 = (
                        int(line[0]),
                        int(line[1]),
                        int(line[2]),
                        int(line[3]),
                    )
                    score = float(line[4])
                    if score > 0.25:
                        if j not in rgbs:
                            r = random.randint(0, 255)
                            g = random.randint(0, 255)
                            b = random.randint(0, 255)
                            rgbs[j] = [r, g, b]
                        rgb = rgbs[j]
                        label = "{}{:.2f}".format(testset.pull_classes()[j], score)
                        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
                        cv2.rectangle(
                            img, (x1, y1 - 15), (x1 + len(label) * 9, y1), rgb, -1
                        )
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
            label = "MutualGuide ({}x{}) : {:.2f}ms on {}".format(
                args.image_size,
                args.image_size,
                detect_time * 1000,
                torch.cuda.get_device_name(0),
            )
            cv2.rectangle(img, (0, 0), (0 + len(label) * 9, 20), [0, 0, 0], -1)
            cv2.putText(
                img,
                label,
                (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            filename = "vis/{}/{}.jpg".format(args.dataset, i)
            cv2.imwrite(filename, img)

        # logging
        if i == 10:
            _t["im_detect"].clear()
            _t["im_nms"].clear()
        if i % math.floor(num_images / 10) == 0 and i > 0:
            print(
                "[{}/{}]Time results: detect={:.2f}ms, nms={:.2f}ms,".format(
                    i, num_images, detect_time * 1000, nms_time * 1000
                )
            )

    # evaluation
    testset.evaluate_detections(all_boxes)
