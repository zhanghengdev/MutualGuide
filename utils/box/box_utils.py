#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax) """

    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h) """

    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def jaccard(box_a, box_b):
    """ Compute the jaccard overlap of two sets of boxes """

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def centerness(box_a, box_b):
    """ Calculate centerness score of center points of box_b according to box_a """
    
    A = box_a.size(0)
    B = box_b.size(0)
    ac_boxes = box_b.unsqueeze(0).expand(A, B, 4)
    gt_boxes = box_a.unsqueeze(1).expand(A, B, 4)
    left_right = torch.stack((ac_boxes[:,:,0]-gt_boxes[:,:,0],     # x-x1
                              gt_boxes[:,:,2]-ac_boxes[:,:,0]), 2) # x2-x
    left_right = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
    left_right[left_right < 0] = 0        # points outside gt boxes
    top_bottom = torch.stack((ac_boxes[:,:,1]-gt_boxes[:,:,1],     # y-y1
                              gt_boxes[:,:,3]-ac_boxes[:,:,1]), 2) # y2-y
    top_bottom = (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    top_bottom[top_bottom < 0] = 0        # points outside gt boxes
    centerness = torch.min(left_right, top_bottom)
    gt_sizes = torch.sqrt((gt_boxes[:,:,2]-gt_boxes[:,:,0]) * (gt_boxes[:,:,3]-gt_boxes[:,:,1]))
    thresh = ac_boxes[:,:,2] / 2.0
    thresh[thresh == thresh.min()] = 0.0
    centerness[gt_sizes <= thresh] = 0.0
    thresh = ac_boxes[:,:,3] * 2.0
    thresh[thresh == thresh.max()] = 1.0
    centerness[gt_sizes >= thresh] = 0.0
    return centerness


def get_foreground(truths, priors, mask, idx, multi_anchor=True):

    if multi_anchor:
        overlaps = jaccard(truths, point_form(priors))
    else:
        overlaps = torch.sqrt(jaccard(truths, point_form(priors)) * centerness(truths, priors))
    (best_truth_overlap, _) = overlaps.max(0)
    best_truth_overlap[best_truth_overlap >= 0.5] = 1.0
    best_truth_overlap[best_truth_overlap < 0.5] = 0.0
    mask[idx] = best_truth_overlap  # [num_priors] jaccord for each prior

def get_foreground2(truths, priors, mask, idx):

    overlaps = centerness(truths, priors)   # Shape: [num_obj, num_priors]
    (best_truth_overlap, _) = overlaps.max(0)
    best_truth_overlap[best_truth_overlap > 0] = 1.0
    mask[idx] = best_truth_overlap  # [num_priors] jaccord for each prior

def match(truths, priors, labels, loc_t, conf_t, overlap_t, idx, multi_anchor=True):
    """ Match each prior box with the ground truth box """

    overlaps = jaccard(truths, point_form(priors)) if multi_anchor else centerness(truths, priors)
    (best_truth_overlap, best_truth_idx) = overlaps.max(0)
    (best_prior_overlap, best_prior_idx) = overlaps.max(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 1)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]


def mutual_match(truths, priors, regress, classif, labels, loc_t, conf_t, overlap_t, pred_t, idx, multi_anchor=True, sigma=2.0):
    """Classify to regress and regress to classify, Mutual Match for label assignement """

    num_obj = truths.size()[0]

    """topk = 10 if multi_anchor else 5
    reg_overlaps = jaccard(truths, decode(regress, priors))
    classif = classif.sigmoid().t()[labels - 1, :]
    reg_overlaps = reg_overlaps ** ((sigma - classif) / sigma)
    reg_overlaps[reg_overlaps != reg_overlaps.max(dim=0, keepdim=True)[0]] = 0.0

    for reg_overlap in reg_overlaps:
        num_pos = max(1, torch.topk(reg_overlap, topk, largest=True)[0].sum().int())
        pos_mask = torch.topk(reg_overlap, num_pos, largest=True)[1]
        reg_overlap[pos_mask] += 3.0

    (best_truth_overlap, best_truth_idx) = reg_overlaps.max(dim=0)
    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    pred_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]
    return"""

    topk = 15 if multi_anchor else 5
    reg_overlaps = jaccard(truths, decode(regress, priors))
    pred_classifs = jaccard(truths, point_form(priors)) if multi_anchor else centerness(truths, priors)
    classif = classif.sigmoid().t()[labels - 1, :]
    pred_classifs = pred_classifs ** ((sigma - classif) / sigma)
    reg_overlaps[reg_overlaps != reg_overlaps.max(dim=0, keepdim=True)[0]] = 0.0
    pred_classifs[pred_classifs != pred_classifs.max(dim=0, keepdim=True)[0]] = 0.0

    for (reg_overlap, pred_classif) in zip(reg_overlaps, pred_classifs):
        num_pos = max(1, torch.topk(reg_overlap, topk, largest=True)[0].sum().int())
        pos_mask = torch.topk(reg_overlap, num_pos, largest=True)[1]
        reg_overlap[pos_mask] += 3.0

        num_pos = max(1, torch.topk(pred_classif, topk, largest=True)[0].sum().int())
        pos_mask = torch.topk(pred_classif, num_pos, largest=True)[1]
        pred_classif[pos_mask] += 3.0

    ## for classification ###
    (best_truth_overlap, best_truth_idx) = reg_overlaps.max(dim=0)
    overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior
    ## for regression ###
    (best_truth_overlap, best_truth_idx) = pred_classifs.max(dim=0)
    pred_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
    loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4]


def encode(matched, priors, variances=[0.1, 0.2]):
    """ Encode from the priorbox layers to ground truth boxes """

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    targets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
    return targets


def decode(loc, priors, variances=[0.1, 0.2]):
    """ Decode locations from predictions using priors """

    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], 
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(dets, thresh=0.5):
    """ Python version Non maximun suppression """

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
