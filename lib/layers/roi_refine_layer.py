# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from utils.bbox_transform import bbox_transform_inv, clip_boxes


def roi_refine_layer(rpn_rois, bbox_pred, im_size):
    """Returns final rois given (subsampled) rpn rois and predicted bbox deltas.

    Args:
        bbox_pred: binary scores of shape (N, 4)
        rpn_rois: binary scores of shape (N, 4)

    Returns:
        rois: (N, 5)
    """
    boxes = bbox_transform_inv(rpn_rois[:, 1:5], bbox_pred)
    boxes = clip_boxes(boxes, im_size)

    # Only support single image as input
    batch_inds = np.zeros((boxes.shape[0], 1), dtype=np.float32)
    rois = np.hstack((batch_inds, boxes.astype(np.float32, copy=False)))

    return rois
