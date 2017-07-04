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
from datasets.cityscapes.cityscapesscripts.labels import NUM_TRAIN_CLASSES
from boxes.bbox_transform import bbox_transform_inv, clip_boxes

from layers.generate_level_anchors import generate_level_anchors
from layers.anchor_target_layer import anchor_target_layer
from layers.proposal_target_layer import proposal_target_layer
from layers.roi_refine_layer import roi_refine_layer
from layers.mask_util import color_mask


# Set to True to visualize mask targets.
# Set to False to verify bbox targets for negative and positive examples.
GT_BOXES_ONLY = True


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

        out_dir = 'out/tests/proposal_target_layer/'
        shutil.rmtree(out_dir, ignore_errors=True)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

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

            anchor_target_layer_labels, _, _, _ = \
                anchor_target_layer(gt_boxes_np, [ih_np, iw_np], anchors, 3)
            # Select 2000 "rois"
            positive = np.where(anchor_target_layer_labels == 1)[0]
            negative = np.where(anchor_target_layer_labels == 0)[0]
            ignored = np.where(anchor_target_layer_labels == -1)[0]
            num = len(positive) + len(negative)
            roi_num = cfg.TRAIN.RPN_POST_NMS_TOP_N
            ignored = np.random.choice(
                np.arange(len(ignored)), size=roi_num-num, replace=False)
            if GT_BOXES_ONLY:
                indices = []
            else:
                indices = np.concatenate([positive, negative, ignored], axis=0)
            input_rois = clip_boxes(anchors[indices, :], [ih_np, iw_np])
            input_rois = np.hstack([np.zeros([len(indices), 1]), input_rois])
            input_scores = np.arange(len(indices))

            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
                crops = \
                    proposal_target_layer(input_rois, input_scores, gt_boxes_np,
                                          np.expand_dims(gt_masks_np, axis=3),
                                          NUM_TRAIN_CLASSES)
            assert np.sum(rois[:, 0]) == 0
            bg = np.where(labels == 0)[0]
            fg = np.where(labels != 0)[0]
            print('fg: {}, bg: {}, target mean: {}'.format(len(fg), len(bg), np.mean(bbox_targets)))

            cls_scores = np.zeros([len(rois), NUM_TRAIN_CLASSES])
            cls_scores[np.arange(len(rois)), labels.astype(int)] = 1.0
            rois = roi_refine_layer(rois, cls_scores, bbox_targets, [ih_np, iw_np])
            bboxes = rois[:, 1:]
            assert np.sum(rois[:, 0]) == 0

            if GT_BOXES_ONLY:
                filled = color_mask(rois, labels, crops, ih_np, iw_np)
                print(np.max(image_np), np.max(filled))
                image_np += 0.5 * filled

            im = Image.fromarray(image_np.astype(np.uint8))
            imd = ImageDraw.Draw(im)
            # green: positive proposal targets, label should not be 'unlabeled'
            # red: negative proposal targets, label should be 'unlabeled'
            for k in range(fg.shape[0]):
                i = fg[k]
                imd.rectangle(bboxes[i, :], outline='green')
                clas = int(labels[i])
                label = trainId2label[clas]
                x0, y0, x1, y1 = bboxes[i, :]
                imd.text(((x0 + x1) / 2, y1), label.name, fill='green')
                weights = np.array([0.0] * 4 * NUM_TRAIN_CLASSES)
                weights[4*clas:4*(clas+1)] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
                assert np.sum(bbox_inside_weights[i, :] - weights) == 0
                weights[4*clas:4*(clas+1)] = 1.0
                assert np.sum(bbox_outside_weights[i, :] - weights) == 0
            for k in range(bg.shape[0]):
                i = bg[k]
                imd.rectangle(bboxes[i, :], outline='red')
                weights = np.array([0.0] * 4 * NUM_TRAIN_CLASSES)
                label = trainId2label[labels[i]]
                x0, y0, x1, y1 = bboxes[i, :]
                imd.text(((x0 + x1) / 2, y1), label.name, fill='red')
                assert np.sum(bbox_targets[i, :]) == 0
                assert np.sum(bbox_outside_weights[i, :] - weights) == 0
                assert np.sum(bbox_outside_weights[i, :] - weights) == 0

            im.save(os.path.join(out_dir, '{}.png'.format(img_id_np)))

        sess.close()
