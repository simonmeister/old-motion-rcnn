# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers

from nets.network import Network
from nets.slim_resnet import resnet_utils
from nets.slim_resnet.resnet_v1 import resnet_v1, bottleneck
from model.config import cfg


def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    data_format = 'NCHW' if cfg.RESNET.USE_NCHW else 'NHWC'
    batch_norm_params = {
        # NOTE 'is_training' is set appropriately inside of the resnet if we pass it to it:
        # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
        # 'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': cfg.RESNET.BN_TRAIN,
        'data_format': data_format,
        'fused': True
    }

    with arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            data_format=data_format,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.max_pool2d, resnet_utils.conv2d_same],
                       data_format=data_format):
            with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc


def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the first unit.
      All other units have stride=1.
      Note that the default slim implementation places the stride in the last unit,
      which is less memory efficient and a deviation from the resnet paper.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }] + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1))


def resnet_v1_50(inputs,
                 is_training=None,
                 scope='resnet_v1_50'):
  """Unlike the slim default we use a stride of 2 in the last block."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
  ]
  return resnet_v1(
      inputs,
      blocks,
      num_classes=None,
      is_training=is_training,
      global_pool=False,
      include_root_block=True,
      scope=scope)


def to_nchw(tensor):
    """Converts default tensorflow NHWC format to NCHW for efficient conv2d.
    Note that NCHW support requires a few small changes to the slim resnet
    implementation.
    """
    if cfg.RESNET.USE_NCHW:
        return tf.transpose(tensor, [0, 3, 1, 2])
    return tensor


def from_nchw(tensor):
    """Converts NCHW format back to default tensorflow NHWC."""
    if cfg.RESNET.USE_NCHW:
        return tf.transpose(tensor, [0, 2, 3, 1])
    return tensor


class resnetv1(Network):
    def __init__(self, *args, **kwargs):
        self._resnet_scope = 'resnet_v1_50'
        Network.__init__(self, *args, **kwargs)

    def _build_pyramid(self, end_points):
        end_points_map = {
            'C2': 'resnet_v1_50/block1/unit_3/bottleneck_v1',
            'C3': 'resnet_v1_50/block2/unit_4/bottleneck_v1',
            'C4': 'resnet_v1_50/block3/unit_6/bottleneck_v1',
            'C5': 'resnet_v1_50/block4/unit_3/bottleneck_v1',
        }
        pyramid = []
        with tf.variable_scope('pyramid'):
            C5 = end_points[end_points_map['C5']]
            P5 = slim.conv2d(C5, 256, [1, 1], stride=1, scope='P5')
            is_nchw = cfg.RESNET.USE_NCHW
            P6 = P5[:, :, ::2, ::2] if is_nchw else P5[:, ::2, ::2, :]
            pyramid = [P6, P5]

            for c in range(4, 1, -1):
                this_C = end_points[end_points_map['C{}'.format(c)]]
                prev_P = pyramid[-1]

                up_shape = tf.shape(this_C)
                up_shape = [up_shape[2], up_shape[3]] if is_nchw else [up_shape[1], up_shape[2]]
                prev_P_up = tf.image.resize_bilinear(from_nchw(prev_P), up_shape,
                                                     name='C{}/upscale'.format(c))
                this_C_adapted = slim.conv2d(this_C, 256, [1,1], stride=1,
                                             scope='C{}'.format(c))

                this_P = tf.add(to_nchw(prev_P_up), this_C_adapted,
                                name='C{}/add'.format(c))
                this_P = slim.conv2d(this_P, 256, [3,3], stride=1,
                                     scope='C{}/refine'.format(c))
                pyramid.append(this_P)
        pyramid = [from_nchw(level) for level in pyramid]
        return pyramid

    def _mask_head(self, roi_crops):
        m = to_nchw(roi_crops)
        for _ in range(4):
            m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
        # to 28x28
        m = slim.conv2d_transpose(m, 256, 2, stride=2, padding='VALID',
                                  activation_fn=tf.nn.relu)
        m = slim.conv2d(m, 1, [1, 1], stride=1, padding='VALID',
                        activation_fn=None)
        return from_nchw(m)

    def _fully_connected_roi_head(self, roi_crops, is_training):
        # to 7x7
        roi_crops = slim.max_pool2d(roi_crops, [2, 2], padding='SAME')
        head = slim.flatten(roi_crops)
        head = slim.fully_connected(head, 1024, activation_fn=tf.nn.relu)
        head = slim.dropout(head, keep_prob=0.5, is_training=is_training)
        head = slim.fully_connected(head, 1024, activation_fn=tf.nn.relu)
        head = slim.dropout(head, keep_prob=0.5, is_training=is_training)
        return head

    def build_network(self, is_training=True):
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv4, end_points = resnet_v1_50(
                to_nchw(self._image),
                is_training=is_training,
                scope=self._resnet_scope)
            pyramid = self._build_pyramid(end_points)

        self._act_summaries.append(net_conv4)
        with tf.variable_scope('RCNN'):
            # build the anchors for the image
            self._build_anchors(pyramid)

            level_outputs = []
            for i, level in enumerate(pyramid):
                level_name = 'P{}'.format(self._pyramid_indices[i])
                with tf.variable_scope(level_name):
                    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
                        rpn = slim.conv2d(to_nchw(level), 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          scope='rpn')

                        rpn_logits = slim.conv2d(rpn, self._num_anchors * 2, [1, 1],
                                                 weights_initializer=initializer,
                                                 padding='VALID',
                                                 activation_fn=None,
                                                 scope='rpn_logits')

                        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1],
                                                    weights_initializer=initializer,
                                                    padding='VALID',
                                                    activation_fn=None,
                                                    scope='rpn_bbox_pred')

                    rpn_logits = tf.reshape(from_nchw(rpn_logits), [-1, 2])
                    rpn_scores = tf.nn.softmax(rpn_logits, dim=1, name='rpn_scores')
                    rpn_bbox_pred = tf.reshape(from_nchw(rpn_bbox_pred), [-1, 4])

                self._act_summaries.append(rpn)
                level_outputs.append((rpn_logits, rpn_scores, rpn_bbox_pred))

            # flattened per-anchor tensors
            rpn_logits, rpn_scores, rpn_bbox_pred = [
                tf.concat(o, axis=0) for o in zip(*level_outputs)]

            if is_training:
                rois, roi_scores = self._proposal_layer(rpn_scores, rpn_bbox_pred, 'rois')
                rpn_labels = self._anchor_target_layer('anchor')
                # Try to have a determinestic order for the computing graph, for reproducibility
                with tf.control_dependencies([rpn_labels]):
                    rois, roi_scores = self._proposal_target_layer(rois, roi_scores, 'rpn_rois')
                # roi_scores is now a single number (the positive score after softmax)
            else:
                if cfg.TEST.MODE == 'nms':
                    rois, roi_scores = self._proposal_layer(rpn_scores, rpn_bbox_pred, 'rois')
                elif cfg.TEST.MODE == 'top':
                    rois, roi_scores = self._proposal_top_layer(rpn_scores, rpn_bbox_pred, 'rois')
                else:
                    raise NotImplementedError

            # all 14x14 roi crops
            roi_crops = self._crop_rois_from_pyramid(rois, pyramid, 'roi_crops')
            fc_roi_features = self._fully_connected_roi_head(roi_crops, is_training)

            cls_logits = slim.fully_connected(fc_roi_features, self._num_classes,
                                             weights_initializer=initializer,
                                             activation_fn=None,
                                             scope='cls_logits')
            cls_scores = tf.nn.softmax(cls_logits, dim=1, name='cls_scores')
            classes = tf.argmax(cls_scores, axis=1)

            bbox_pred = slim.fully_connected(fc_roi_features,
                                             self._num_classes * 4,
                                             weights_initializer=initializer_bbox,
                                             activation_fn=None, scope='bbox_pred')

            if not is_training:
                if cfg[self._mode].BBOX_REG:
                    rois = self._roi_refine_layer(rois, cls_scores, bbox_pred,
                                                  'refined_rois')
                rois, roi_scores, cls_scores = self._mask_layer(rois, roi_scores, cls_scores,
                                                                'testing_rois')
                # crop with changed rois (subsampled and/or refined)
                roi_crops = self._crop_rois_from_pyramid(rois, pyramid, name='roi_crops')

            with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
                mask_logits = self._mask_head(roi_crops)
            mask_scores = tf.sigmoid(mask_logits, name='mask_scores')
            masks = tf.to_float(mask_scores >= 0.5, name='masks')

        self._predictions['rpn_logits'] = rpn_logits
        self._predictions['rpn_scores'] = rpn_scores
        self._predictions['rpn_bbox_pred'] = rpn_bbox_pred
        self._predictions['cls_scores'] = cls_scores
        self._predictions['bbox_pred'] = bbox_pred
        self._predictions['rois'] = rois
        self._predictions['mask_logits'] = mask_logits
        self._predictions['mask_scores'] = masks
        self._predictions['masks'] = masks
        self._predictions['classes'] = classes
        self._predictions['scores'] = roi_scores

        if is_training:
            self._predictions['cls_logits'] = cls_logits

        self._score_summaries.update(self._predictions)

        return rois, cls_scores, bbox_pred
