# --------------------------------------------------------
# Tensorflow Mask R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Charles Shang
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np

from model.config import cfg
from utils.cython_bbox import bbox_overlaps


def mask_target_layer(rois, roi_scores, cls_scores, gt_boxes, cfg_key):
    """Returns rois for mask_branch, each with a gt box assigned for cropping mask targets.

    Note that one ground truth box can be assigned to multiple rois.
    The final targets will still differ in how the ground truth masks
    are cropped with the roi bounding box.

    Args:
        rois: (T, 5)
        roi_scores: (T,)
        cls_scores: (T, num_classes)
        gt_boxes: (G, 5)

    Returns:
        rois: [[batch_id, x1, y1, x2, y2], ...] of shape (M, 5)
        roi_scores: (M,)
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    mask_thresh = cfg[cfg_key].MASK_THRESH
    masks_top_n = cfg[cfg_key].MASKS_TOP_N

    # num_rois x num_gt_boxes
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  # gt mask batch index per-roi
    max_overlaps = overlaps[np.arange(len(gt_assignment)), gt_assignment] # per-roi

    keep_inds = np.where(max_overlaps >= mask_thresh)[0]
    num_pos = int(min(keep_inds.size, masks_top_n))
    if keep_inds.size > 0 and num_pos < keep_inds.size:
        keep_inds = np.random.choice(keep_inds, size=num_pos, replace=False)

    # TODO negative samples (1:3 pos:neg)
    # or use as many positives as there are and the rest negative? makes more sense here, as some images have more and we want to use these
    #
    # select rois with least overlaps, return BEST (or worst??) MATCHING bounding box for them
    # (ideally, the crop will then be all zeros)
    # => during testing, we can return exactly N = 64 mask_branch_rois
    # see anchor_target_layer

    #negative_gt_assignment = overlaps.argmin(axis=1)
    #min_overlaps = overlaps[np.arange(len(negative_gt_assignment)),
    #                        negative_gt_assignment]
    #order = min_overlaps.ravel().argsort()[::-1]
    #keep_neg_inds = np.where(min_overlaps < mask_thresh)[0]
    #num_neg = int(min(keep_neg_inds.size, masks_top_n - num_pos))
    #if keep_neg_inds.size > 0 and num_masks < keep_neg_inds.size:
    #    keep_neg_inds = np.random.choice(keep_neg_inds, size=num_masks, replace=False)

    rois = np.hstack((gt_assignment[keep_inds], rois[keep_inds, :]))
    roi_scores = roi_scores[keep_inds]
    cls_scores = cls_scores[keep_inds]
    return rois, roi_scores, cls_scores
