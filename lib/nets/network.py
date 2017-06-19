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
from layers.proposal_top_layer import proposal_top_layer
from layers.anchor_target_layer import anchor_target_layer
from layers.proposal_target_layer import proposal_target_layer
from layers.generate_level_anchors import generate_level_anchors
from layers.assign_boxes import assign_boxes
from layers.mask_target_layer import mask_target_layer
from layers.mask_layer import mask_layer

from model.config import cfg


class Network(object):
    def __init__(self, batch_size=1,
                image, im_info, gt_boxes, gt_masks,
                mode, num_classes,
                anchor_scales=[8], anchor_ratios=[0.5, 1, 2]):
        self._pyramid_strides = [4, 8, 16, 32, 64]
        self._pyramid_indices = [2, 3, 4, 5, 6]
        self._batch_size = batch_size
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

    ###########################################################################
    # Mask R-CNN layers
    ###########################################################################

    def _crop_rois(self, image, rois, name, resized_height, resized_width):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name='batch_id'), [1])
            # Get the normalized coordinates of bboxes
            height, width = tf.shape(image)[1:3]
            x1 = tf.slice(rois, [0, 1], [-1, 1], name='x1') / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name='y1') / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name='x2') / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name='y2') / height
            # Won't be backpropagated to boxes anyway, but to save time # TODO verify
            boxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            crops = tf.image.crop_and_resize(image, rois,
                                             tf.to_int32(batch_ids),
                                             [resized_height, resized_width],
                                             name='crops')
        return crops

    def _assign_boxes(self, boxes):
        assignments = tf.py_func(assign_boxes,
                                 [boxes, self._pyramid_indices[0], self._pyramid_indices[-1]],
                                 [tf.int32], name='assign_boxes')
        return assignments

    def _crop_rois_from_pyramid(self, rois, pyramid, name):
        """rois is (N, 5), where first entry is batch"""
        with tf.variable_scope(name) as scope:
            boxes = tf.slice(rois, [0, 1], [-1, -1])
            level_assignments = self._assign_boxes(boxes)
            reordered_roi_crops = []
            reordered_indices = []

            for i, level in zip(self._pyramid_indices, pyramid):
                indices = tf.where(level_assignments == i)
                reordered_rois = tf.gather_nd(rois, indices)
                roi_crops = self._crop_rois(level, reordered_rois,
                                            resized_height=14, resized_width=14,
                                            name='roi_crops_{}'.format(i))
                reordered_roi_crops.append(roi_crops)
                reordered_indices.append(indices)

            reordered_roi_crops = tf.concat(reordered_roi_crops, axis=0)
            reordered_indices = tf.concat(reordered_indices, axis=0)
            num_rois = tf.shape(boxes)[0]
            roi_crops_shape = tf.constant([num_rois, 14, 14, 256])
            roi_crops = tf.scatter_nd(reordered_indices, reordered_roi_crops, roi_crops_shape)
        return roi_crops

    def _build_anchors(pyramid):
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
            anchors = tf.py_func(generate_level_anchors,
                                 [height, width, stride,
                                  self._anchor_scales, self._anchor_ratios],
                                  [tf.float32], name='generate_level_anchors')
        return anchors

    def _mask_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            mask_branch_rois = tf.py_func(
                mask_layer,
                [rois, roi_scores, self._mode],
                [tf.float32])
        return mask_branch_rois

    def _mask_target_layer(self, rois, name):
        with tf.variable_scope(name) as scope:
            gt_masks = self._gt_masks
            mask_branch_rois = tf.py_func(
                mask_target_layer,
                [rois, self._gt_boxes, self._mode],
                [tf.float32])

            gt_crops = self._crop_rois(gt_masks, mask_branch_rois,
                                       resized_height=28, resized_width=28,
                                       name='gt_crops')
            self._mask_targets['mask_targets'] = mask_targets
            self._score_summaries.update(self._mask_targets)
        return mask_branch_rois

    ###########################################################################
    # Faster R-CNN layers
    ###########################################################################

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name='to_int32')
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name='to_int32')
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    ###########################################################################
    # Utilities
    ###########################################################################

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _reshape_layer(self, input, num_dim, name): #TODO why not reshape to [-1 , 2]?
        input_shape = tf.shape(input)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_nchw = tf.transpose(input, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped_nchw = tf.reshape(to_nchw,
                                       tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            reshaped = tf.transpose(reshaped_nchw, [0, 2, 3, 1])
            return reshaped

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,
                        sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
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
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets,
                                                rpn_bbox_inside_weights, rpn_bbox_outside_weights,
                                                sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions['cls_score']
            label = tf.reshape(self._proposal_targets['labels'], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets,
                                            bbox_inside_weights, bbox_outside_weights)

            # RCNN, mask loss
            mask_targets = self._mask_targets['mask_targets']
            masks = self._predictions['masks']
            loss_mask = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=mask_targets, logits=masks))

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            self._losses['mask_loss'] = loss_mask

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + loss_mask
            self._losses['total_loss'] = loss

            self._event_summaries.update(self._losses)

        return loss

    def build_network(self, is_training=True):
        raise NotImplementedError

    def create_architecture(input_batch,
                            mode, num_classes,
                            anchor_scales=[8], anchor_ratios=[0.5, 1, 2]):

        self._image = input_batch[0]
        self._im_info = input_batch[1]
        self._gt_boxes = input_batch[2]
        self._gt_masks = input_batch[3]
        # TODO handle examples with len(gt_boxes) == 0 by removing them during preprocesing/loading or in the tfrecords

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        
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
            self.build_network(training)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions['bbox_pred'] *= stds
            self._predictions['bbox_pred'] += means
        else:
            self._add_losses()

        val_summaries = []
        with tf.device('/cpu:0'):
            # TODO rather use _add_image_summary on predictions - we know that the loader is OK
            # this means we have to select the predicted class's bboxes from bbox_pred
            # and convert masks to full image painting
            # BEST TO SAVE BOTH!
            val_summaries.append(self._add_image_summary(
                self._image, self._gt_boxes, self._gt_masks))
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

    ###########################################################################
    # Summaries
    ###########################################################################

    def _add_image_summary(self, image, boxes, masks):
        # add back mean
        image += cfg.PIXEL_MEANS # TODO
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)
        # TODO overlay mask paintings
        return tf.summary.image('ground_truth', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)
