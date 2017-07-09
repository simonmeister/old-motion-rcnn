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
from boxes.bbox_transform import bbox_transform_inv, clip_boxes


def roi_refine_layer(rpn_rois, cls_scores, bbox_pred, im_size):
    """Returns final rois given (subsampled) rpn rois by applying the
    deltas of the highest scoring classes.

    Args:
        rpn_rois: (N, 5)
        cls_scores: (N, num_classes)
        bbox_pred: (N, num_classes * 4)

    Returns:
        rois: (N, 5)
    """
    top_classes = np.argmax(cls_scores, axis=1)
    num_classes = cls_scores.shape[1]
    num_rois = cls_scores.shape[0]

    bbox_pred = np.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_pred = bbox_pred[np.arange(num_rois), top_classes, :]

    boxes = bbox_transform_inv(rpn_rois[:, 1:], bbox_pred)
    boxes = clip_boxes(boxes, im_size)

    # Only support single image as input
    batch_inds = np.zeros((boxes.shape[0], 1), dtype=np.float32)
    rois = np.hstack((batch_inds, boxes.astype(np.float32, copy=False)))

    #import pdb; pdb.set_trace()
    return rois
