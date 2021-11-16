#!/usr/bin/python
# -*- coding: utf-8 -*-

import torchvision
from .box_utils import decode

def Detect(predictions, prior, scale, eval_thresh=0.01, nms_thresh=0.5):
    """ Detect layer at test time """

    (loc, conf) = predictions['loc'], predictions['conf']
    assert loc.size(0) == 1,  'ERROR: Batch size = {} during evaluation'.format(loc.size(0))

    (loc, conf) = loc.squeeze(0), conf.squeeze(0)
    decoded_boxes = decode(loc, prior).clamp(min=0, max=1)
    decoded_boxes *= scale  # scale each detection back up to the image
    conf_scores = conf.sigmoid()
    
    keep = conf_scores.max(1)[0] > eval_thresh
    decoded_boxes=decoded_boxes[keep]
    conf_scores=conf_scores[keep]
    
    keep = torchvision.ops.nms(decoded_boxes, conf_scores.max(1)[0], iou_threshold=nms_thresh)
    decoded_boxes=decoded_boxes[keep]
    conf_scores=conf_scores[keep]

    (decoded_boxes, conf_scores) = decoded_boxes.cpu().numpy(), conf_scores.cpu().numpy()
    return (decoded_boxes, conf_scores)

