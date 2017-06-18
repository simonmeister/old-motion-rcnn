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

import datasets.reader as reader
from datasets.cityscapes.labels import id2label

# from libs.layers import ROIAlign

# resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS


with tf.Graph().as_default():
    file_pattern = '/home/smeister/datasets/motion-rcnn/records/cityscapes/val/*.tfrecord'
    tfrecords = glob.glob(file_pattern)

    image, ih, iw, gt_boxes, gt_masks, gt_classes, num_instances, img_id = \
        reader.read(tfrecords)

    # image, gt_boxes, gt_masks = \
    #  preprocess_coco.preprocess_image(image, gt_boxes, gt_masks)

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    # boxes = [[100, 100, 200, 200],
    #         [50, 50, 100, 100],
    #         [100, 100, 750, 750],
    #         [50, 50, 60, 60]]
    # boxes = np.zeros((0, 4))
    # boxes = tf.constant(boxes, tf.float32)
    # feat = ROIAlign(image, boxes, False, 16, 7, 7)
    sess.run(init_op)

    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        for i in range(30):
            image_np, ih_np, iw_np, gt_boxes_np, gt_masks_np, num_instances_np, img_id_np = \
                sess.run([image, ih, iw, gt_boxes, gt_masks, num_instances, img_id])
            img_id_np = img_id_np.decode('utf8')
            print('image_id: {}, instances: {}, shape: {}'.format(img_id_np, num_instances_np, image_np.shape))
            # image_np = 256 * (image_np * 0.5 + 0.5)
            # image_np = image_np.astype(np.uint8)
            image_np = np.squeeze(image_np)
            im = Image.fromarray(image_np)
            imd = ImageDraw.Draw(im)
            for i in range(gt_boxes_np.shape[0]):
                label = id2label[gt_boxes_np[i, 4]]
                color = 'rgb({},{},{})'.format(*label.color)
                pos = gt_boxes_np[i, :4]
                x0, y0, x1, y1 = pos
                imd.rectangle(pos, outline=color)
                imd.text(((x0 + x1) / 2, y1), label.name, fill=color)
            im.save('data/' + str(img_id_np) + '.png')
            # print (gt_boxes_np)
        sess.close()
