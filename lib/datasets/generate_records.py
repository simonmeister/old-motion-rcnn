from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .cityscapes.tfrecords import create_records as create_cityscapes

tf.app.flags.DEFINE_string(
    'dataset_name', 'cityscapes',
    'The name of the dataset to convert.')

tf.app.flags.DEFINE_string(
    'datasets_dir', '',
    'Data source directory.')

tf.app.flags.DEFINE_bool(
    'shuffle', True,
    'Whether to shuffle files before storing them in shards. Should be True for training data.')

tf.app.flags.DEFINE_string(
    'records_dir', '/home/smeister/datasets/motion-rcnn/records',
    'Where to store tfrecords for use during training.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if FLAGS.dataset_name == 'cityscapes':
        create_cityscapes(FLAGS.records_dir, FLAGS.datasets_dir, FLAGS.dataset_split_name)
    else:
        raise ValueError(
            'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)


if __name__ == '__main__':
    tf.app.run()
