# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from utils.bbox_transform import bbox_transform_inv, clip_boxes
import numpy.random as npr


def proposal_top_layer(rpn_scores, rpn_bbox_pred, im_info, anchors, num_anchors):
    """Given predicted objectness and bbox deltas, returns the bboxes and scores of top
    scoring roi proposals.

    This variant selects the top N region proposals without using non-maximal suppression.

    Args:
        rpn_scores: binary scores of shape (A, 2)
        rpn_bbox_pred: of shape (A, 4)
        anchors: (A, 4), all anchors
        num_anchors: number of anchors per position

    Returns:
        rois: (N, 5), where N << A, with first column zero
        scores: (N,), objectness (positive) score
    """
    rpn_top_n = cfg.TEST.RPN_TOP_N
    im_info = im_info[0]

    scores = rpn_scores[:, 0:1]

    length = scores.shape[0]
    if length < rpn_top_n:
        # Random selection, maybe unnecessary and loses good proposals
        # But such case rarely happens
        top_inds = npr.choice(length, size=rpn_top_n, replace=True)
    else:
        top_inds = scores.argsort(0)[::-1]
        top_inds = top_inds[:rpn_top_n]
        top_inds = top_inds.reshape(rpn_top_n, )

    # Do the selection here
    anchors = anchors[top_inds, :]
    rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
    scores = scores[top_inds]

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

    # Clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob, scores
