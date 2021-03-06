# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np

from boxes.nms_wrapper import nms
from model.config import cfg


def test_layer(rois, roi_scores, cls_scores, cfg_key):
    """Returns (score-ordered) rois and scores for mask branch.

    Args:
        rois: (T, 5)
        roi_scores: (T,)
        cls_scores: (T, num_classes)

    Returns:
        rois: [[batch_id, x1, y1, x2, y2], ...] of shape (M, 5)
        roi_scores: (M,)
        cls_scores: (M, num_classes)
    """
    topN = cfg.TEST.POST_NMS_TOP_N
    nms_thresh = cfg.TEST.NMS_THRESH

    # Non-maximal suppression
    keep = nms(np.hstack((rois[:, 1:], np.reshape(roi_scores, [-1, 1]))),
               nms_thresh)
    rois = rois[keep, :]
    roi_scores = roi_scores[keep]
    cls_scores = cls_scores[keep, :]
    # Pick top scoring after nms
    order = roi_scores.ravel().argsort()[::-1]
    order = order[:topN]
    rois = rois[order, :]
    roi_scores = roi_scores[order]
    cls_scores = cls_scores[order, :]

    # Only support single image as input
    batch_inds = np.zeros((rois.shape[0], 1), dtype=np.float32)
    rois = np.hstack((batch_inds, rois[:, 1:]))

    return rois, roi_scores, cls_scores
