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
import matplotlib.pyplot as plt

import _init_paths
from model.config import cfg
from datasets import reader
from datasets.preprocess import preprocess_example
from datasets.cityscapes.cityscapesscripts.labels import trainId2label


with tf.Graph().as_default():
    file_pattern = cfg.TFRECORD_DIR + '/cityscapes/train/*.tfrecord'
    tfrecords = glob.glob(file_pattern)

    with tf.device('/cpu:0'):
        example = reader.read(tfrecords)
        example = preprocess_example(example, is_training=True, normalize=False)

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    sess.run(init_op)

    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        for i in range(30):
            example_np = sess.run(example)
            img_id_np = example_np['id'].decode('utf8')
            image_np = example_np['image']
            gt_boxes_np = example_np['boxes']
            gt_masks_np = example_np['masks']
            depth_np = example_np['depth']
            #print(np.count_nonzero(np.isinf(depth_np)))
            #import matplotlib.pyplot as plt
            depth_mask = tf.to_float(
                tf.logical_and(
                    tf.logical_not(tf.is_inf(depth_np)),
                    depth_np != 0))
            print(sess.run(depth_mask))
            print(sess.run(tf.reduce_sum(depth_np * depth_mask)))
            #plt.imshow(depth_np[:, :, 0], cmap='gray')
            #plt.show()
            num_instances_np = gt_masks_np.shape[0]
            print('image_id: {}, instances: {}, shape: {}'
                  .format(img_id_np, num_instances_np, image_np.shape))
            image_np = np.squeeze(image_np)

            # overlay masks
            for i in range(gt_boxes_np.shape[0]):
                label = trainId2label[gt_boxes_np[i, 4]]
                mask = np.expand_dims(gt_masks_np[i, :, :], 2)
                image_np += 0.5 * mask * np.array(label.color)

            # draw boxes
            im = Image.fromarray(image_np.astype(np.uint8))
            imd = ImageDraw.Draw(im)
            for i in range(gt_boxes_np.shape[0]):
                label = trainId2label[gt_boxes_np[i, 4]]
                color = 'rgb({},{},{})'.format(*label.color)
                pos = gt_boxes_np[i, :4]
                x0, y0, x1, y1 = pos
                imd.rectangle(pos, outline=color)
                imd.text(((x0 + x1) / 2, y1), label.name, fill=color)

            out_dir = 'out/tests/cityscapes/'
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            im.save(os.path.join(out_dir, str(img_id_np) + '.png'))
        sess.close()
