# --------------------------------------------------------
# Tensorflow Mask R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Charles Shang and Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np

from model.config import cfg


def mask_layer(rois, roi_scores, cfg_key):
    """Returns rois for mask_branch.

    Note that one ground truth box can be assigned to multiple rois.
    The final targets will still differ in how the ground truth masks
    are cropped with the roi bounding box.

    Args:
        rois: (T, 5)
        roi_scores: (T,)

    Returns:
        rois: [[batch_id, x1, y1, x2, y2], ...] of shape (M, 5)
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    masks_top_n = cfg[cfg_key].MASKS_TOP_N

    order = scores.ravel().argsort()[::-1]
    if masks_top_n > 0:
        order = order[:masks_top_n]
    rois = rois[order, :]
    rois = rois[keep, :]

    batch_inds = np.zeros((rois.shape[0], 1), dtype=np.float32)
    rois = np.hstack((batch_inds, rois))
    return rois
