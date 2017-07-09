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


def get_batch(dataset_name, split_name, records_root, num_classes,
              batch_size=1, is_training=False, num_threads=4,
              epochs=1):
    file_pattern = dataset_name + '/' + split_name + '/' + '*.tfrecord'
    tfrecords = glob.glob(records_root + '/' + file_pattern)

    example = read(tfrecords, shuffle=is_training, epochs=epochs)
    example = preprocess_example(example, is_training)

    as_list = list(example.values())
    dtypes = [v.dtype for v in as_list]

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
            dtypes=dtypes)
    else:
        capacity = min_after_dequeue * 2

        data_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=dtypes)

    enqueue_op = data_queue.enqueue(as_list)
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * num_threads)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    as_list_deq =  data_queue.dequeue()

    example = dict(zip(example.keys(), as_list_deq))

    example['image'] = tf.expand_dims(example['image'], axis=0)
    example['image'].set_shape([1, None, None, 3])
    size = tf.unstack(tf.shape(example['image']), num=4)[1:3]
    h, w = size
    example['size'] = size
    example['boxes'] = tf.reshape(example['boxes'], [-1, 5])
    example['masks'] = tf.reshape(example['masks'], [-1, h, w, 1])
    if 'depth' in example:
        example['depth'] = tf.expand_dims(example['depth'], axis=0)
        example['depth'].set_shape([1, None, None, 1])
    example['num_classes'] = num_classes

    return example
