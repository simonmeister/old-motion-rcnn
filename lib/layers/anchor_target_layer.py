# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from boxes.cython_bbox import bbox_overlaps
from boxes.bbox_transform import bbox_transform


def anchor_target_layer(gt_boxes, im_size, all_anchors, num_anchors):
    """Returns targets for all rpn anchor predictions.

    Unlike the anchor target layer in the original Faster R-CNN,
    boxes intersecting the image boundaries are not removed.

    Args:
        gt_boxes: (G, 5)
        all_anchors: (A, 4), A = B * H * W * num_anchors
        num_anchors: number of anchors per position

    Returns:
        labels: (A,), -1 for ignore, 0 for negative and 1 for positive example
        bbox_targets: (A, 4)
        bbox_inside_weights: (A, 4), row zero if label == -1 (no example), else one
        bbox_outside_weights: (A, 4), contribution of each example row to final loss (0 if ignore)
    """
    total_anchors = len(all_anchors)

    # allow boxes to sit over the edge by a small amount
    #_allowed_border = 0

    # TODO can we remove this?
    #inds_inside = np.where(
    #    (all_anchors[:, 0] >= -_allowed_border) &
    #    (all_anchors[:, 1] >= -_allowed_border) &
    #    (all_anchors[:, 2] < im_size[1] + _allowed_border) &  # width
    #    (all_anchors[:, 3] < im_size[0] + _allowed_border)  # height
    #)[0]
    inds_inside = np.arange(total_anchors)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is don't care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes, (A, G)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    # TODO Maybe select the same amount of boxes from each level?
    #   We could pass a list of anchor arrays instead to do that..
    #   Could it otherwise learn to predict more positive for larger examples?
    #   Or we could select at least one negative box of the same level for any pos. box of a level
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    #print("{} anchors, {} boxes, {} ==1, {} ==0, {} fg, {} bg"
    #      .format(len(all_anchors), len(gt_boxes), np.sum(labels == 1),
    #       np.sum(labels == 0), len(fg_inds), len(bg_inds)))

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        # TODO for this to work, we need to modify Network._smooth_l1_loss
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
