#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


class SeqBoxMatcher(object):
    """ Late fusion for video object detection """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        semantic_thresh: float = 0.1,
        seq_thresh: int = 2,
    ) -> None:
        super(SeqBoxMatcher, self).__init__()
        self.iou_thresh = iou_thresh
        self.semantic_thresh = semantic_thresh
        self.seq_thresh = seq_thresh
        self.reset()
    
    def reset(
        self,
    ) -> None:
        self.reference = []

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list:

        def matrix_iou(a,b):
            """ return iou of a and b, numpy version for data augenmentation """

            lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
            rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
            area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
            area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
            area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
            return area_i / (area_a[:, np.newaxis] + area_b - area_i)

        new_boxes = np.hstack(
                (boxes, scores, np.ones((boxes.shape[0],1)))
            ).astype(np.float32)
        
        if new_boxes.shape[0] == 0:
            self.reset()
            return boxes, scores
        
        if len(self.reference)>0:
            iou_matrix = matrix_iou(new_boxes[:,:4], self.reference[:,:4])
            iou_matrix[iou_matrix < self.iou_thresh] = 0
            prob_matrix = np.dot(new_boxes[:,4:-1], self.reference[:,4:-1].T)
            prob_matrix[prob_matrix < self.semantic_thresh] = 0
            sim_matrix = iou_matrix*prob_matrix
            pairs = {}
            for index in sim_matrix.reshape(-1).argsort()[::-1]:
                i, j = np.unravel_index(index, sim_matrix.shape)
                if sim_matrix[i,j] == 0:
                    break
                if (i in pairs) or (j in list(pairs.values())):
                    continue
                pairs[i] = j
            for i,j in pairs.items():
                new_boxes[i,4:-1] = np.add(
                        np.dot(
                                self.reference[j,4:-1], self.reference[j,-1]/(self.reference[j,-1]+1)
                            ),
                        np.dot(
                                new_boxes[i,4:-1], 1/(self.reference[j,-1]+1)
                            ),
                        )
                new_boxes[i,-1] += self.reference[j,-1]
        self.reference = new_boxes
        mask = self.reference[:,-1] > self.seq_thresh
        return self.reference[mask, :4], self.reference[mask, 4:-1]
