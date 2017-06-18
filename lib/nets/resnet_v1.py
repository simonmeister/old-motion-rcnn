# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

from nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg


# TODO generate anchors ONCE

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        # NOTE 'is_training' here does not work because inside resnet it gets reset:
        # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': cfg.RESNET.BN_TRAIN,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class resnetv1(Network):
    def __init__(self, batch_size=1, num_layers=50):
        Network.__init__(self, batch_size=batch_size)
        self._num_layers = num_layers
        self._resnet_scope = 'resnet_v1_%d' % num_layers

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def _build_base(self): # TODO understand why this is done
        with tf.variable_scope(self._resnet_scope, self._resnet_scope):
            net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def _build_pyramid(self, end_points, end_points_map):
        pyramid = []
        with tf.variable_scope('pyramid'):
            C5 = end_points[end_points_map['C5']]
            pyramid = [slim.conv2d(C5, 256, [1, 1], stride=1, scope='P5')]

            for c in range(4, 1, -1):
                this_C, prev_P = pyramid[-1], end_points[pyramid_map['C{}'.format(c)]]

                up_shape = tf.shape(this_C)
                prev_P_up = tf.image.resize_bilinear(prev_P, [up_shape[1], up_shape[2]],
                                                     name='C{}/upscale'.format(c))
                this_C_adapted = slim.conv2d(this_C, 256, [1,1], stride=1,
                                             scope='C{}'.format(c))

                this_P = tf.add(prev_P_up, this_C_adapted, name='C{}/add'.format(c))
                this_P = slim.conv2d(s, 256, [3,3], stride=1, scope='C{}/refine'.format(c))
                pyramid.append(this_P)
        pyramid = pyramid[::-1]
        return pyramid

    def _mask_branch(self, roi_crops):
        m = roi_crops
        for _ in range(4):
            m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        # to 28x28
        m = slim.conv2d_transpose(m, 256, 2, stride=2, padding='VALID', activation_fn=tf.nn.relu)
        m = slim.conv2d(m, 1, [1, 1], stride=1, padding='VALID', activation_fn=None)
        return m

    def _fully_connected_roi_head(self, roi_crops):
        # to 7x7
        roi_crops = slim.max_pool2d(roi_crops, [2, 2], padding='SAME')
        head = slim.flatten(roi_crops)
        head = slim.fully_connected(refine, 1024, activation_fn=tf.nn.relu)
        head = slim.dropout(refine, keep_prob=0.5, is_training=is_training)
        head = slim.fully_connected(refine, 1024, activation_fn=tf.nn.relu)
        head = slim.dropout(refine, keep_prob=0.5, is_training=is_training)
        return head

    def build_network(self, sess, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        bottleneck = resnet_v1.bottleneck
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            blocks = [
                resnet_utils.Block('block1', bottleneck,
                                   [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                resnet_utils.Block('block2', bottleneck,
                                   [(512, 128, 1)] * 3 + [(512, 128, 2)]),
                # Use stride-1 for the last conv4 layer
                resnet_utils.Block('block3', bottleneck,
                                   [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
            ]
            end_points_map = {
                'C2': 'resnet_v1_50/block1/unit_3/bottleneck_v1',
                'C3': 'resnet_v1_50/block2/unit_4/bottleneck_v1',
                'C4': 'resnet_v1_50/block3/unit_6/bottleneck_v1',
                'C5': 'resnet_v1_50/block4/unit_3/bottleneck_v1',
            }
        else:
            # other numbers are not supported
            raise NotImplementedError

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net = self._build_base()
            net_conv4, end_points = resnet_v1.resnet_v1(
                net, blocks,
                global_pool=False,
                include_root_block=False,
                scope=self._resnet_scope)
            pyramid = self._build_pyramid(end_points, end_points_map)

        self._act_summaries.append(net_conv4) # TODO
        self._layers['head'] = net_conv4 # TODO
        with tf.variable_scope(self._resnet_scope, self._resnet_scope):
            # build the anchors for the image
            self._build_anchors(pyramid)

            level_outputs = []
            for level in pyramid:
                rpn = slim.conv2d(level, 512, [3, 3],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  scope="rpn_conv/3x3")
                self._act_summaries.append(rpn) # TODO adapt?
                rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                            weights_initializer=initializer,
                                            padding='VALID', activation_fn=None, scope='rpn_cls_score')
                # change it so that the score has 2 as its channel size
                rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape') # TODO understand why this is done!
                rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
                rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
                rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                            weights_initializer=initializer,
                                            padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
                level_outputs.append((rpn_cls_score_reshape,
                                      rpn_cls_prob_reshape,
                                      rpn_cls_prob,
                                      rpn_bbox_pred))

            rpn_cls_score_reshape, \
            rpn_cls_prob_reshape, \
            rpn_cls_prob, \
            rpn_bbox_pred = \
                [tf.concat(o, axis=0) for o in zip(*level_outputs)]

            if is_training:
                rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
                # Try to have a determinestic order for the computing graph, for reproducibility
                with tf.control_dependencies([rpn_labels]):
                    rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois") # roi_scores
                    mask_branch_rois = self._mask_target_layer(rois, 'mask_branch_rois')
            else:
                if cfg.TEST.MODE == 'nms':
                    rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                elif cfg.TEST.MODE == 'top':
                    rois, roi_scores = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                else:
                    raise NotImplementedError

                mask_branch_rois = self._mask_layer(rois, 'mask_branch_rois)

            # all 14x14 roi crops
            roi_crops = self._crop_rois_from_pyramid(rois, pyramid, "roi_crops")
            fc_roi_head = self.fully_connected_roi_head(roi_crops)

            cls_score = slim.fully_connected(fc_roi_head, self._num_classes,
                                             weights_initializer=initializer,
                                             trainable=is_training,
                                             activation_fn=None,
                                             scope='cls_score')
            cls_prob = self._softmax_layer(cls_score, "cls_prob")

            bbox_pred = slim.fully_connected(fc_roi_head,
                                             self._num_classes * 4,
                                             weights_initializer=initializer_bbox,
                                             trainable=is_training,
                                             activation_fn=None, scope='bbox_pred')

            # TODO (later) use refined boxes for crops for mask branch at test time
            # need to use a subset of proposal_layer.py to create simple bbox_transform wrapper - see FastMaskRCNN
            # mask head # TODO summaries??
            mask_branch_roi_crops = self._crop_rois_from_pyramid(mask_branch_rois, pyramid,
                                                                 name="roi_crops")
            masks = self._mask_branch(mask_branch_roi_crops)

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois
        self._predictions["masks"] = masks

        self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred
