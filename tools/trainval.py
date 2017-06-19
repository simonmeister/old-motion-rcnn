from __future__ import absolute_import, division, print_function

import tensorflow as tf

from datasets.cityscapes.labels import NUM_TRAIN_CLASSES
from datasets.factory import get_dataset
from model.train_val import Trainer
from model.config import cfg, cfg_from_file, cfg_from_list, write_cfg_to_file
from nets.resnet_v1 import resnetv1

tf.app.flags.DEFINE_string(
    'dataset', 'cityscapes',
    'The dataset to train.')

tf.app.flags.DEFINE_string(
    'epochs', '',
    'Epoch or comma-separated list of epochs to train.')

tf.app.flags.DEFINE_string(
    'lrs', '',
    'Learning rate or comma-separated list of learning rates to train.')

tf.app.flags.DEFINE_string(
    'train_split', 'train',
    'Dataset split to train on.')

tf.app.flags.DEFINE_string(
    'val_split', 'val',
    'Dataset split to validate on.')

tf.app.flags.DEFINE_string(
    'tag', '',
    'Model tag.')

FLAGS = tf.app.flags.FLAGS


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--data', dest='dataset',
                      help='dataset to train on',
                      default='cityscapes', type=str)
  parser.add_argument('--split', dest='train_split',
                      help='dataset split to train on',
                      default='train', type=str)
  parser.add_argument('--valsplit', dest='val_split',
                      help='dataset split to train on',
                      default='val', type=str)
  parser.add_argument('--ex', dest='experiment_name',
                      help='name of experiment',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='backbone network',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


class Dataset()
  pass


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    ex_cfg_file = os.path.join(cfg.CONFIG_DIR, args.experiment_name + '.yml')

    if os.path.isfile(ex_cfg_file):
        cfg_from_file(ex_cfg_file)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    if not os.path.isdir(cfg.CONFIG_DIR):
        os.makedirs(cfg.CONFIG_DIR)
    write_cfg_to_file(ex_cfg_file)

    assert args.dataset in ['cityscapes']
    assert args.train_split in ['train', 'val', 'mini']
    assert args.val_split in ['val', 'mini']

    ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, args.experiment_name)
    log_dir = os.path.join(cfg.LOG_DIR, args.experiment_name)

    dataset = Dataset()
    dataset.num_classes = NUM_TRAIN_CLASSES

    dataset.get_train_batch = lambda: get_example(
        args.dataset, args.train_split, cfg.TFRECORD_DIR,
        is_training=True, batch_size=cfg.TRAIN.BATCH_SIZE)

    dataset.get_test_batch = lambda: get_example(
        args.dataset, FLAGS.train_split, cfg.TFRECORDS_DIR,
        is_training=False, batch_size=cfg.TRAIN.BATCH_SIZE)

    network = resnetv1(batch_size=cfg.TRAIN.BATCH_SIZE)
    trainer = Trainer(network, dataset,
                      ckpt_dir=ckpt_dir, tbdir=log_dir)
    trainer.train_val(zip(cfg.TRAIN.EPOCHS, cfg.TRAIN.LEARNING_RATES))
