#!/usr/bin/env python
# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import sys
import os
import glob

import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import tensorflow as tf
from pprint import pprint
import shutil

import _init_paths
from model.config import cfg
from datasets import reader
from datasets.preprocess import preprocess_example
from datasets.cityscapes.cityscapesscripts.labels import trainId2label
from layers.generate_level_anchors import generate_level_anchors
from layers.anchor_target_layer import anchor_target_layer
from boxes.bbox_transform import bbox_transform_inv


with tf.Graph().as_default():
    file_pattern = cfg.TFRECORD_DIR + '/cityscapes/train/*.tfrecord'
    tfrecords = glob.glob(file_pattern)

    with tf.device('/cpu:0'):
        image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = reader.read(tfrecords)
        image, gt_boxes, gt_masks = preprocess_example(image, gt_boxes, gt_masks, is_training=True,
                                                       normalize=False)
        ih, iw = tf.unstack(tf.shape(image))[:2]

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    sess.run(init_op)

    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():

        out_dir = 'out/tests/anchor_target_layer/'
        shutil.rmtree(out_dir, ignore_errors=True)

        for i in range(10):
            image_np, ih_np, iw_np, gt_boxes_np, gt_masks_np, num_instances_np, img_id_np = \
                sess.run([image, ih, iw, gt_boxes, gt_masks, num_instances, img_id])
            img_id_np = img_id_np.decode('utf8')
            print('image_id: {}, instances: {}, shape: {}'.format(img_id_np, num_instances_np, image_np.shape))
            image_np = np.squeeze(image_np)

            anchors = []
            for feat_stride in [64, 32, 16, 8, 4]:
                anchor_boxes = generate_level_anchors(ih_np / feat_stride, iw_np / feat_stride,
                                                      feat_stride=feat_stride)
                anchors.append(anchor_boxes)
            anchors = np.concatenate(anchors, axis=0)

            labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                anchor_target_layer(gt_boxes_np, [ih_np, iw_np], anchors, 3)

            bboxes = bbox_transform_inv(anchors, bbox_targets)
            positive = np.where(labels == 1)[0]
            negative = np.where(labels == 0)[0]
            ignored = np.where(labels == -1)[0]
            num = len(positive) + len(negative)

            im = Image.fromarray(image_np.astype(np.uint8))
            imd = ImageDraw.Draw(im)
            # blue: ground truth boxes
            # white: regression target boxes for positive anchors (recovered from deltas)
            # green: positive anchors
            # red: negative anchors
            for i in range(gt_boxes_np.shape[0]):
                imd.rectangle(gt_boxes_np[i, :], outline='blue')
            for k in range(positive.shape[0]):
                i = positive[k]
                imd.rectangle(bboxes[i, :])
                imd.rectangle(anchors[i, :], outline='green')
                assert np.sum(bbox_inside_weights[i, :] - np.array([1.0] * 4)) == 0
                assert np.sum(bbox_outside_weights[i, :] - np.array([1.0 / num] * 4)) == 0
            for k in range(negative.shape[0]):
                i = negative[k]
                imd.rectangle(anchors[i, :], outline='red')
                assert np.sum(bbox_inside_weights[i, :] - np.array([0.0] * 4)) == 0
                assert np.sum(bbox_outside_weights[i, :] - np.array([1.0 / num] * 4)) == 0
            #for k in range(ignored.shape[0]):
            #    i = ignored[k]
            #    assert np.sum(bbox_inside_weights[i, :] - np.array([0.0] * 4)) == 0
            #    assert np.sum(bbox_outside_weights[i, :] - np.array([0.0] * 4)) == 0

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            im.save(os.path.join(out_dir, '{}.png'.format(img_id_np)))

        sess.close()
