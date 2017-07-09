# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layers.proposal_layer import proposal_layer
from layers.anchor_target_layer import anchor_target_layer
from layers.proposal_target_layer import proposal_target_layer
from layers.generate_level_anchors import generate_level_anchors
from layers.assign_to_levels import assign_to_levels
from layers.test_layer import test_layer
from layers.roi_refine_layer import roi_refine_layer
from layers.mask_util import color_mask

from model.config import cfg


class Network(object):
    def __init__(self, example, is_training):
        self._pyramid_strides = [64, 32, 16, 8, 4]
        self._pyramid_indices = [6, 5, 4, 3, 2]
        self._batch_size = 1
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._mask_targets = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}

        self._input = example
        self._num_classes = example['num_classes']

        self._mode = 'TRAIN' if is_training else 'TEST'
        self._anchor_scales = cfg.ANCHOR_SCALES,
        self._num_scales = len(self._anchor_scales)

        self._anchor_ratios = cfg.ANCHOR_RATIOS
        self._num_ratios = len(self._anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self.build_network(is_training)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if not is_training and cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self.__input['size']))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self.__input['size']))
            self._predictions['bbox_pred'] *= stds
            self._predictions['bbox_pred'] += means

        if is_training:
            self._add_losses()

        with tf.device('/cpu:0'):
            self.summary_image = self._get_summary_image(
                self._input['image'],
                self._predictions['rois'],
                self._predictions['classes'],
                self._predictions['mask_scores'])
            for key, var in self._event_summaries.items():
                tf.summary.scalar(key, var)
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)

    ###########################################################################
    # Mask R-CNN layers
    ###########################################################################

    def _crop_rois(self, image, rois, name, resized_height, resized_width, batch_ids=None):
        with tf.variable_scope(name) as scope:
            if batch_ids is None:
                batch_ids = rois[:, 0]
            # Get the normalized coordinates of bboxes
            height = tf.to_float(self._input['size'][0])
            width = tf.to_float(self._input['size'][1])
            x1 = rois[:, 1] / width
            y1 = rois[:, 2] / height
            x2 = rois[:, 3] / width
            y2 = rois[:, 4] / height
            boxes = tf.stack([y1, x1, y2, x2], axis=1)
            crops = tf.image.crop_and_resize(image, boxes,
                                             tf.to_int32(batch_ids),
                                             [resized_height, resized_width],
                                             name='crops')
        return crops

    def _assign_to_levels(self, boxes):
        assignments, = tf.py_func(assign_to_levels,
                                  [boxes, self._input['size'], len(self._pyramid_strides),
                                   self._pyramid_strides[-1]],
                                  [tf.int32], name='assign_to_levels')

        assignments.set_shape([None])

        return assignments

    def _crop_rois_from_pyramid(self, rois, pyramid, name):
        """rois is (N, 5), where first entry is batch"""
        with tf.variable_scope(name) as scope:
            level_assignments = self._assign_to_levels(rois[:, 1:])
            reordered_roi_crops = []
            reordered_indices = []

            for i, level in enumerate(pyramid):
                indices = tf.where(tf.equal(level_assignments, i))[:, 0]
                reordered_rois = tf.gather(rois, indices)
                roi_crops = self._crop_rois(level, reordered_rois,
                                            resized_height=14, resized_width=14,
                                            name='roi_crops_{}'.format(i))
                reordered_roi_crops.append(roi_crops)
                reordered_indices.append(indices)

            reordered_roi_crops = tf.concat(reordered_roi_crops, axis=0)
            reordered_indices = tf.to_int32(tf.concat(reordered_indices, axis=0))
            num_rois = tf.unstack(tf.shape(rois))[0]
            roi_crops_shape = tf.stack([num_rois, 14, 14, 256], axis=0)
            reordered_indices = tf.expand_dims(reordered_indices, axis=1)
            roi_crops = tf.scatter_nd(reordered_indices, reordered_roi_crops, roi_crops_shape)
        return roi_crops

    def _build_anchors(self, pyramid):
        anchors = []
        for level, stride in zip(pyramid, self._pyramid_strides):
            level_anchors = self._generate_level_anchors(level, stride)
            anchors.append(level_anchors)
        anchors = tf.concat(anchors, axis=0)
        self._anchors = anchors
        return anchors

    def _generate_level_anchors(self, level, stride):
        with tf.variable_scope('ANCHOR_' + str(stride)) as scope:
            height, width = tf.unstack(tf.shape(level))[1:3]
            anchors, = tf.py_func(generate_level_anchors,
                                  [height, width, stride,
                                   self._anchor_scales, self._anchor_ratios],
                                   [tf.float32], name='generate_level_anchors')

            anchors.set_shape([None, 4])

        return anchors

    def _roi_refine_layer(self, rois, cls_scores, bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, = tf.py_func(
                roi_refine_layer,
                [rois, cls_scores, bbox_pred, self._input['size']],
                [tf.float32])

            rois.set_shape([None, 5])

        return rois

    def _test_layer(self, rois, roi_scores, cls_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, cls_scores = tf.py_func(
                test_layer,
                [rois, roi_scores, cls_scores, self._mode],
                [tf.float32, tf.float32, tf.float32])

            rois.set_shape([None, 5])
            roi_scores.set_shape([None])
            cls_scores.set_shape([None])

        return rois, roi_scores, cls_scores

    ###########################################################################
    # Faster R-CNN layers
    ###########################################################################

    def _anchor_target_layer(self, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [self._input['boxes'], self._input['size'],
                 self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels.set_shape([None])
            rpn_bbox_targets.set_shape([None, 4])
            rpn_bbox_inside_weights.set_shape([None, 4])
            rpn_bbox_outside_weights.set_shape([None, 4])

            rpn_labels = tf.to_int32(rpn_labels, name='to_int32')
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_layer(self, rpn_scores, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_logits = tf.py_func(proposal_layer,
                                          [rpn_scores, rpn_bbox_pred, self._input['size'],
                                           self._mode, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_logits.set_shape([None])

        return rois, rpn_logits

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, \
                bbox_outside_weights, mask_targets = tf.py_func(
                    proposal_target_layer,
                    [rois, roi_scores,
                     self._input['boxes'], self._input['masks'], self._num_classes],
                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                     tf.float32])

            rois.set_shape([None, 5])
            roi_scores.set_shape([None])
            labels.set_shape([None])
            bbox_targets.set_shape([None, self._num_classes * 4])
            bbox_inside_weights.set_shape([None, self._num_classes * 4])
            bbox_outside_weights.set_shape([None, self._num_classes * 4])

            #gt_crops = self._crop_rois(self._gt_masks, rois,
            #                           batch_ids=gt_assignments,
            #                           resized_height=28, resized_width=28,
            #                           name='gt_crops')

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name='to_int32')
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            self._proposal_targets['mask_targets'] = mask_targets

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    ###########################################################################
    # Utilities
    ###########################################################################
    def _l1_loss(self, diff, valid):
        # diff may also contain inf or nan, so we use tf.where instead of multiplying
        # with a mask tensor
        diff = tf.where(valid, diff, tf.zeros(tf.shape(diff)))
        count = tf.reduce_sum(tf.to_float(valid))
        loss = tf.reduce_sum(tf.abs(diff)) / count
        return loss

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,
                        sigma=1.0, dim=[1]):
        """Computes smooth l1 loss between bbox_pred and bbox_targets.

        There are two usages:

        1. All examples are weighted the same, there are no ignored terms:
            - bbox_inside_weights is 0 at negative examples, a positive constant otherwise
            - bbox_outside_weights is 0 at negative examples, 1 otherwise
                => all diffs are averaged and thus negative examples contribute to the loss
                via the normalization
            - dim should be [1] so that we sum over 4 target numbers but average over examples

        2. Manual weighting and support for ignored terms:
            - bbox_inside_weights is zero at negative and ignored examples, a positive constant
            otherwise
            - bbox_outside_weights is zero at ignored examples and non-zero otherwise
                => used to scale losses before summing them up. E.g. for uniform weighting of
                pos. and neg., set bbox_outside_weights to 1 / num_non_ignored at non-ignored
                examples.
            - dim should be [0, 1] to sum along all axes
        """
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        #smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        smoothL1_sign = tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                        + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss') as scope:
            # RPN, class loss
            rpn_logits = self._predictions['rpn_logits']
            rpn_labels = self._anchor_targets['rpn_labels']
            rpn_select = tf.where(tf.not_equal(rpn_labels, -1))
            rpn_logits = tf.gather(rpn_logits, rpn_select)
            rpn_labels = tf.gather(rpn_labels, rpn_select)
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=rpn_logits, labels=rpn_labels))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets,
                                                rpn_bbox_inside_weights, rpn_bbox_outside_weights,
                                                sigma=sigma_rpn, dim=[0, 1])

            # RCNN, class loss
            cls_logits = self._predictions['cls_logits']
            label = self._proposal_targets['labels']
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=cls_logits, labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets,
                                            bbox_inside_weights, bbox_outside_weights,
                                            dim=[1])

            # RCNN, mask loss
            mask_targets = self._proposal_targets['mask_targets']
            mask_logits = self._predictions['mask_logits']
            loss_mask = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=mask_targets, logits=mask_logits))

            # Depth prediction, supervised loss
            loss_depth = 0
            if 'depth' in self._input:
                gt_depth = self._input['depth']
                depth_pred = self._predictions['depth_pred']
                if cfg.TRAIN.INVERSE_DEPTH:
                    depth_target = 1. / gt_depth
                    valid = gt_depth > 0
                else:
                    depth_target = gt_depth
                    valid = tf.logical_and(gt_depth > 0, gt_depth < np.inf)
                diff = depth_pred - depth_target
                loss_depth = self._l1_loss(diff, valid)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            self._losses['mask_loss'] = loss_mask
            self._losses['depth'] = loss_depth

            loss = 0
            if cfg.TRAIN.RPN:
                loss += rpn_cross_entropy + rpn_loss_box
            if cfg.TRAIN.RCNN:
                loss += cross_entropy + loss_box + loss_mask
            if cfg.TRAIN.MOTION:
                if cfg.TRAIN.SUPERVISE_DEPTH:
                    loss += loss_depth
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    def build_network(self, is_training=True):
        raise NotImplementedError

    ###########################################################################
    # Summaries
    ###########################################################################

    def _color_mask(self, rois, classes, masks, height, width):
        im, = tf.py_func(
            color_mask,
            [rois, classes, masks, height, width],
            [tf.float32])

        im.set_shape([None, None, 3])

        return im

    def _get_summary_image(self, image, rois, classes, masks):
        # add back mean
        image += cfg.PIXEL_MEANS / 255.0
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [batch, x1, y1, x2, y2] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(rois, axis=1)
        boxes = tf.stack([cols[2] / height,
                          cols[1] / width,
                          cols[4] / height,
                          cols[3] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)
        #color_mask = self._color_mask(rois, classes, masks, # TODO add again
        #                              *tf.unstack(tf.shape(image))[1:3])
        #image = image + 0.4 * color_mask
        return image

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)
