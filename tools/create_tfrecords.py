from __future__ import absolute_import, division, print_function

import tensorflow as tf

import _init_paths
from datasets.cityscapes.tfrecords import create_records as create_cityscapes
from model.config import cfg

tf.app.flags.DEFINE_string(
    'data', 'cityscapes',
    'The name of the dataset to create tfrecords for.')

tf.app.flags.DEFINE_string(
    'split', 'mini',
    'Dataset split to create. One of (train,val,test,mini)')

tf.app.flags.DEFINE_bool(
    'shuffle', True,
    'Whether to shuffle files before storing them in shards. Should be True for training splits.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.data == 'cityscapes':
        create_cityscapes(cfg.TFRECORD_DIR, cfg.DATA_DIR, FLAGS.split)
    else:
        raise ValueError('invalid dataset')


if __name__ == '__main__':
    tf.app.run()
