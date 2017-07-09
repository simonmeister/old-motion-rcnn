# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Charles Shang
# --------------------------------------------------------
import os
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


def read(tfrecord_filenames, shuffle=False, epochs=1):
    if not isinstance(tfrecord_filenames, list):
        tfrecord_filenames = [tfrecord_filenames]
    filename_queue = tf.train.string_input_producer(
        tfrecord_filenames, num_epochs=epochs, shuffle=shuffle,
        capacity=len(tfrecord_filenames))

    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/id': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'label/num_instances': tf.FixedLenFeature([], tf.int64),
            'label/masks': tf.FixedLenFeature([], tf.string),
            'label/boxes': tf.FixedLenFeature([], tf.string),
            'label/depth': tf.FixedLenFeature([], tf.string)
        })
    img_id = features['image/id']
    ih = tf.cast(features['image/height'], tf.int32)
    iw = tf.cast(features['image/width'], tf.int32)
    num_instances = tf.cast(features['label/num_instances'], tf.int32)
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    imsize = tf.size(image)
    image = tf.cond(tf.equal(imsize, ih * iw), \
                    lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
                    lambda: tf.reshape(image, (ih, iw, 3)))

    boxes = tf.decode_raw(features['label/boxes'], tf.float32)
    boxes = tf.reshape(boxes, [num_instances, 5])
    masks = tf.decode_raw(features['label/masks'], tf.uint8)
    masks = tf.reshape(masks, [num_instances, ih, iw])

    # TODO what if gt has no depth? make more flexible..
    # if 'label/depth' in features:
    depth = tf.decode_raw(features['label/depth'], tf.float32)
    depth = tf.reshape(depth, (ih, iw, 1))

    return {
        'image': image,
        'boxes': boxes,
        'masks': masks,
        'depth': depth,
        'id': img_id
    }
