#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.backends.cudnn as cudnn


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """

    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, 
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """

    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, 
                       boxes[:, 2:] - boxes[:, :2]), 1)

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    '''Compute the jaccard overlap of two sets of boxes.  The jaccard overlap\n    is simply the intersection over union of two boxes.  Here we operate on\n    ground truth boxes and default boxes.\n    E.g.:\n        A \xe2\x88\xa9 B / A \xe2\x88\xaa B = A \xe2\x88\xa9 B / (area(A) + area(B) - A \xe2\x88\xa9 B)\n    Args:\n        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]\n        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]\n    Return:\n        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]\n    '''

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def centerness(box_a, box_b):
    """ Calculate centerness score of center points of box_b
    according to box_a.
      Box_a should be (x, y, w, h)
      Box_b should be (x, y, w, h)
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) centerness score, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    anchor_centers = box_b[:, :2].unsqueeze(0).expand(A, B, 2)  # Shape [A, B, 2]
    gt_boxes = box_a.unsqueeze(1).expand(A, B, 4)
    left_right = torch.stack((anchor_centers[:,:,0]-gt_boxes[:,:,0],     # x-x1
                              gt_boxes[:,:,2]-anchor_centers[:,:,0]), 2) # x2-x
    left_right = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
    left_right[left_right < 0] = 0        # points outside gt boxes
    top_bottom = torch.stack((anchor_centers[:,:,1]-gt_boxes[:,:,1],     # y-y1
                              gt_boxes[:,:,3]-anchor_centers[:,:,1]), 2) # y2-y
    top_bottom = (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    top_bottom[top_bottom < 0] = 0        # points outside gt boxes
    return torch.min(left_right, top_bottom)


def get_foreground(truths, priors, mask, idx):

    # overlaps = centerness(truths, priors)   # Shape: [num_obj, num_priors]
    # (best_truth_overlap, best_truth_idx) = overlaps.max(0)
    # best_truth_overlap[best_truth_overlap > 0] = 1.0
    # mask[idx] = best_truth_overlap  # [num_priors] jaccord for each prior

    overlaps = jaccard(truths, point_form(priors))
    (best_truth_overlap, best_truth_idx) = overlaps.max(0)
    best_truth_overlap[best_truth_overlap >= 0.5] = 1.0
    best_truth_overlap[best_truth_overlap < 0.5] = 0.0
    mask[idx] = best_truth_overlap  # [num_priors] jaccord for each prior


def match(truths, priors, labels, loc_t, conf_t, overlap_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        overlap_t: (tensor) Tensor to be filled w/ iou score for each priors.
        overlap_t: (tensor) Tensor to be filled w/ match object idx for each priors.
        idx: (int) current batch index
    """

    overlaps = jaccard(truths, point_form(priors))
    (best_truth_overlap, best_truth_idx) = overlaps.max(0)
    (best_prior_overlap, best_prior_idx) = overlaps.max(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 1)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]


def mutual_match(truths, priors, regress, classif, labels, loc_t, conf_t, overlap_t, pred_t, idx):
    """Classify to regress and regress to classify, Mutual Match for label assignement.
    Args:
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4].
        regress: (tensor) Regression prediction, Shape: [num_priors, 4].
        classif: (tensor) Classification prediction, Shape: [num_priors, num_classes].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        overlap_t: (tensor) Tensor to be filled w/ iou score for each priors.
        overlap_t: (tensor) Tensor to be filled w/ pred score for each priors.
        idx: (int) current batch index
    """

    num_obj = truths.size()[0]
    acr_overlaps = jaccard(truths, point_form(priors))
    reg_overlaps = jaccard(truths, decode(regress, priors))
    pred_classifs = classif.sigmoid().t()[labels - 1, :]
    sigma = 2.0
    pred_classifs = acr_overlaps ** ((sigma - pred_classifs) / sigma)
    acr_overlaps[torch.arange(num_obj), acr_overlaps.max(1)[1]] = 1.0
    reg_overlaps[torch.arange(num_obj), reg_overlaps.max(1)[1]] = 1.0
    pred_classifs[torch.arange(num_obj), pred_classifs.max(1)[1]] = 1.0
    acr_overlaps[acr_overlaps != acr_overlaps.max(dim=0, keepdim=True)[0]] = 0.0
    reg_overlaps[reg_overlaps != reg_overlaps.max(dim=0, keepdim=True)[0]] = 0.0
    pred_classifs[pred_classifs != pred_classifs.max(dim=0, keepdim=True)[0]] = 0.0

    for (reg_overlap, pred_classif, acr_overlap) in zip(reg_overlaps, pred_classifs, acr_overlaps):
        num_ign = (acr_overlap >= 0.4).sum()
        num_pos = (acr_overlap >= 0.5).sum()

        ign_mask = torch.topk(reg_overlap, num_ign, largest=True)[1]
        pos_mask = torch.topk(reg_overlap, num_pos, largest=True)[1]
        reg_overlap[ign_mask] = 2.0
        reg_overlap[pos_mask] = 3.0

        pos_mask = torch.topk(pred_classif, num_pos, largest=True)[1]
        pred_classif[pos_mask] = 3.0

    ## for classification ###
    (best_truth_overlap, best_truth_idx) = reg_overlaps.max(dim=0)
    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    ## for regression ###
    (best_truth_overlap, best_truth_idx) = pred_classifs.max(dim=0)
    pred_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]


def encode(matched, priors, variances=[0.1, 0.2]):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    targets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
    return targets


def decode(loc, priors, variances=[0.1, 0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], 
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """

    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def nms(dets, thresh=0.5):
    """Python version Non maximun suppression. 
        See: https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    Args:
        dets (numpy arrays): detected bounding boxes
        thresh (float): iou threshold
        mode (string): iou or ciou
    """

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

