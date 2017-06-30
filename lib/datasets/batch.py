# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
import glob

import tensorflow as tf

from datasets.reader import read
from datasets.preprocess import preprocess_example
from model.config import cfg

# TODO adapt for motion-rcnn after baseline works

def get_batch(dataset_name, split_name, records_root,
              batch_size=1, is_training=False, num_threads=4,
              epochs=1):
    file_pattern = dataset_name + '/' + split_name + '/' + '*.tfrecord'
    tfrecords = glob.glob(records_root + '/' + file_pattern)

    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = read(tfrecords, shuffle=is_training,
                                                                    epochs=epochs)
    image, gt_boxes, gt_masks = preprocess_example(image, gt_boxes, gt_masks, is_training)

    min_after_dequeue = cfg.TRAIN.MIN_EXAMPLES_AFTER_DEQUEUE
    capacity = min_after_dequeue * 2

    # We randomly resize images for avoiding overfitting, so we don't know static
    # width and height and have to use custom batching code.
    # Currently, only batch size of 1 is supported. In the future, we could adapt preprocess
    # to resize a whole batch to a single random size.
    if is_training:
        data_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            dtypes=(image.dtype, gt_boxes.dtype, gt_masks.dtype))
    else:
        capacity = min_after_dequeue * 2

        data_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=(image.dtype, gt_boxes.dtype, gt_masks.dtype))

    enqueue_op = data_queue.enqueue((image, gt_boxes, gt_masks))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * num_threads)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    image, gt_boxes, gt_masks =  data_queue.dequeue()

    image = tf.expand_dims(image, axis=0)
    image.set_shape([1, None, None, 3])
    gt_boxes = tf.expand_dims(gt_boxes, axis=0)
    gt_masks = tf.expand_dims(gt_masks, axis=0)

    return [image, gt_boxes, gt_masks]
