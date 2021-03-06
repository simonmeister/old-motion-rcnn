# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from boxes.bbox_transform import bbox_transform_inv, clip_boxes
from boxes.nms_wrapper import nms


def proposal_layer(rpn_scores, rpn_bbox_pred, im_size, cfg_key, anchors, num_anchors):
    """Given predicted objectness and bbox deltas, returns the bboxes and scores of top
    scoring roi proposals.

    A simplified variant compared to fast/er RCNN.

    Args:
        rpn_scores: binary scores of shape (A, 2)
        rpn_bbox_pred: of shape (A, 4)
        anchors: (A, 4), all anchors, A = B * H * W * num_anchors
        num_anchors: number of anchors per position

    Returns:
        rois: (N, 5), where N << A, with first column zero
        scores: (N,), objectness (positive) score
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    # Get the scores and bounding boxes
    scores = rpn_scores[:, 1]
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_size)

    # TODO add this again?
    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_size[2])
    # keep = _filter_boxes(proposals, min_size * im_size[2])
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, np.reshape(scores, [-1, 1]))),
               nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
