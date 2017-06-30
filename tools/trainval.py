from __future__ import absolute_import, division, print_function

import argparse
import pprint
import sys
import os
import shutil

import tensorflow as tf

import _init_paths
from datasets.cityscapes.cityscapesscripts.labels import NUM_TRAIN_CLASSES
from datasets.batch import get_batch
from model.trainer import Trainer
from model.config import cfg, cfg_from_file, cfg_from_list, write_cfg_to_file
from nets.resnet_v1 import resnetv1


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Motion R-CNN network')
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
                        help='dataset split to evalute on during training',
                        default='val', type=str)
    parser.add_argument('--ex', dest='experiment_name',
                        help='name of experiment',
                        default='default', type=str)
    parser.add_argument('--ow', dest='overwrite',
                        help='overwrite experiment',
                        action='store_true')
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu id to train on',
                        default='0', type=str)
    parser.add_argument('--mode', dest='mode', # TODO
                        help='one of (train, val, trainval)',
                        default='trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


class Dataset():
    pass


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    ex_cfg_file = os.path.join(cfg.CONFIG_DIR, args.experiment_name + '.yml')

    if os.path.isfile(ex_cfg_file) and not args.overwrite:
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

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, args.experiment_name)
    log_dir = os.path.join(cfg.LOG_DIR, args.experiment_name)
    if args.overwrite:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)


    dataset = Dataset()
    dataset.num_classes = NUM_TRAIN_CLASSES

    dataset.get_train_batch = lambda: get_batch(
        args.dataset, args.train_split, cfg.TFRECORD_DIR,
        is_training=True)

    dataset.get_val_batch = lambda: get_batch(
        args.dataset, args.val_split, cfg.TFRECORD_DIR,
        is_training=False)

    trainer = Trainer(resnetv1, dataset,
                      pretrained_model='data/models/resnet_v1_50.ckpt',
                      ckpt_dir=ckpt_dir, tbdir=log_dir)
    if args.mode == 'trainval':
        trainer.train_val(zip(cfg.TRAIN.EPOCHS, cfg.TRAIN.LEARNING_RATES))
    elif args.mode == 'train':
        trainer.train(zip(cfg.TRAIN.EPOCHS, cfg.TRAIN.LEARNING_RATES))
    elif args.mode == 'val':
        trainer.evaluate()
    else:
        raise NotImplementedError("--mode must be one of (trainval, train, val)")
