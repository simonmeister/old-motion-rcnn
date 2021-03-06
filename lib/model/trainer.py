# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import sys
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib import slim

from model.config import cfg
from boxes.timer import Timer
from datasets.cityscapes.cityscapesscripts.evaluate import evaluate_np_preds as evaluate_cs
import datasets.cityscapes.cityscapesscripts.labels as labels
from layers.mask_util import binary_mask


class Trainer(object):
    """A wrapper class for the training process."""

    def __init__(self, network_cls, dataset,
                 ckpt_dir, tbdir, pretrained_model=None):
        self.network_cls = network_cls
        self.dataset = dataset
        self.ckpt_dir = ckpt_dir
        self.tbdir = tbdir
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbdir):
            os.makedirs(self.tbdir)
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.pretrained_model = pretrained_model

    def train_val(self, schedule, val=True):
        for epochs, learning_rate in schedule:
            for epoch in range(epochs):
                self.train_epoch(learning_rate)
                if val:
                    self.evaluate()

    def train(self, schedule):
        # TODO this doesn't work properly yet as we rely on the coord stop signal
        # after each epoch to save checkpoints
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            self._train_epochs(sess, list(schedule))

    def train_epoch(self, learning_rate):
        with tf.Graph().as_default():
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            with tf.Session(config=tfconfig) as sess:
                self._train_epochs(sess, [(1, learning_rate)])

    def _train_epochs(self, sess, schedule):
        total_epochs = sum([ep for ep, lr in schedule])
        print("Training for {} epoch(s): {}.".format(total_epochs, schedule))
        with tf.device('/cpu:0'):
            batch = self.dataset.get_train_batch(total_epochs)

        net = self.network_cls(batch, is_training=True)

        train_op, lr_placeholder = self._get_train_op(net)

        saver = tf.train.Saver(max_to_keep=cfg.TRAIN.CHECKPOINTS_MAX_TO_KEEP,
                               keep_checkpoint_every_n_hours=4)

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if ckpt is None:
            assert self.pretrained_model, 'Pre-trained resnet_v1_50 not found. See README.'
            vars_to_restore = slim.get_variables_to_restore(include=['resnet_v1_50'])
            vars_to_restore = [v for v in vars_to_restore if not 'Momentum' in v.name]
            restorer = tf.train.Saver(vars_to_restore)
            print('Loading initial model weights from {}'.format(self.pretrained_model))
            restorer.restore(sess, self.pretrained_model)
            print('Loaded.')
            epoch = 0
        else:
            ckpt_path = ckpt.model_checkpoint_path
            print('Restoring model checkpoint from {}'.format(ckpt_path))
            saver.restore(sess, ckpt_path)
            print('Restored.')
            epoch = int(ckpt_path.split('/')[-1].split('-')[-1]) + 1

        seed = cfg.RNG_INITIAL_SEED + epoch * cfg.RNG_EPOCH_SEED_INCREMENT
        np.random.seed(seed)
        tf.set_random_seed(seed)

        writer = tf.summary.FileWriter(self.tbdir, sess.graph)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        timer = Timer()

        for local_epochs, lr in schedule:
            for _ in range(local_epochs):
                i = 0
                max_i = cfg.TRAIN.EXAMPLES_PER_EPOCH - 1
                try:
                    while not coord.should_stop():
                        timer.tic()
                        feed_dict = {lr_placeholder: lr}

                        run_ops = [
                            net._losses['rpn_cross_entropy'],
                            net._losses['rpn_loss_box'],
                            net._losses['cross_entropy'],
                            net._losses['loss_box'],
                            net._losses['mask_loss'],
                            net._losses['total_loss'],
                            train_op
                        ]

                        if i % cfg.TRAIN.SUMMARY_INTERVAL == 0:
                            run_ops.append(tf.summary.merge_all())

                        run_results = sess.run(run_ops, feed_dict=feed_dict)

                        if i % cfg.TRAIN.SUMMARY_INTERVAL == 0:
                            summary = run_results[-1]
                            run_results = run_results[:-1]
                            writer.add_summary(summary, float(i - 1) + epoch * max_i)

                        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, mask_loss, total_loss, _ \
                            = run_results

                        timer.toc()

                        if i % cfg.TRAIN.DISPLAY_INTERVAL == 0:
                            print('{} [{} / {} at {:.3f} s/batch & lr {}] '
                                  'loss: {:.4f} [RPN cls {:.4f} box {:.4f}] [cls {:.4f} box {:.4f} mask {:.4f}]'
                                  .format(epoch, i, max_i, timer.average_time, lr,
                                          total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box,
                                          mask_loss))
                        i += 1
                except tf.errors.OutOfRangeError:
                    pass

                save_filename = os.path.join(self.ckpt_dir, 'model.ckpt')
                saver.save(sess, save_filename, global_step=epoch)
                epoch += 1

        writer.close()
        coord.request_stop()
        coord.join(threads)
        print('Finished epoch {} and wrote checkpoint.'.format(epoch))

    def _get_train_op(self, net):
        loss = net._losses['total_loss']

        lr_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        tf.summary.scalar('TRAIN/learning_rate', lr_placeholder)
        optimizer = tf.train.MomentumOptimizer(lr_placeholder, cfg.TRAIN.MOMENTUM)

        # Compute the gradients wrt the loss
        gvs = optimizer.compute_gradients(loss)
        # Double the gradient of the bias if set
        if cfg.TRAIN.DOUBLE_BIAS:
            final_gvs = []
            with tf.variable_scope('Gradient_Mult') as scope:
                for grad, var in gvs:
                    scale = 1.
                    if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                        scale *= 2.
                    if not np.allclose(scale, 1.0):
                        grad = tf.multiply(grad, scale)
                    final_gvs.append((grad, var))
            train_op = optimizer.apply_gradients(final_gvs)
        else:
            train_op = optimizer.apply_gradients(gvs)

        return train_op, lr_placeholder

    def evaluate(self):
        with tf.Graph().as_default():
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            with tf.Session(config=tfconfig) as sess:
                self._evaluate_cs(sess)
        # TODO add KITTI 2015 eval

    def _evaluate_cs(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        assert ckpt is not None

        writer = tf.summary.FileWriter(self.tbvaldir)

        ckpt_path = ckpt.model_checkpoint_path
        epoch = int(ckpt_path.split('/')[-1].split('-')[-1])
        with tf.device('/cpu:0'):
            batch = self.dataset.get_val_batch()
        image = batch['image']
        net = self.network_cls(batch, is_training=False)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        print('Loading model checkpoint for evaluation: {:s}'.format(ckpt_path))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print('Loaded.')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        iters = 0
        avg_losses = np.zeros([len(net._losses)])
        pred_np_arrays = []
        summary_images = []
        #depths = []
        try:
            while iters < 3: #not coord.should_stop():
                loss_ops = [v for (k, v) in net._losses]
                pred_ops = [
                    net._predictions['mask_scores'],
                    net._predictions['cls_scores'],
                    net._predictions['scores'],
                    net._predictions['rois']
                ]

                run_results = sess.run(loss_ops + pred_ops + [tf.shape(image), net.summary_image])
                loss_results = run_results[:len(loss_ops)]
                pred_results = run_results[len(loss_ops):-2]
                image_shape_np = run_results[-2]
                summary_image_np = run_results[-1]
                avg_losses += loss_results
                pred_np_arrays.append(pred_results)
                summary_images.append(summary_image_np)
                #depths.append(depth)
                iters += 1

                print("\rPredicted: {}".format(iters), end=' ')
                sys.stdout.flush()
            print('')

        except tf.errors.OutOfRangeError:
            pass

        avg_losses /= iters
        height, width = image_shape_np[1:3]
        pred_lists = []
        for masks, cls_scores, rpn_scores, rois in pred_np_arrays:
            print(np.mean(rois[:, 3] - rois[:, 1]),
                  np.mean(rois[:, 4] - rois[:, 2]),
                  np.mean(masks))
            preds = []
            for i in range(masks.shape[0]):
                train_id = np.argmax(cls_scores[i])
                if train_id == 0:
                    # Ignore background class
                    continue
                print(cls_scores[i], rpn_scores[i], rois[i, 1:])
                pred_dct = {}
                pred_dct['imgId'] = "todo"
                pred_dct['labelID'] = labels.trainId2label[train_id].id
                pred_dct['conf'] = rpn_scores[i]
                mask = binary_mask(rois[i, :], masks[i, :, :], height, width)
                pred_dct['binaryMask'] = mask.astype(np.uint8)
                preds.append(pred_dct)
            pred_lists.append(preds)

        cs_avgs = evaluate_cs(pred_lists)
        max_images = min(len(summary_images), cfg.TEST.MAX_SUMMARY_IMAGES)
        for i, im in enumerate(summary_images[:max_images]):
            tf.summary.image('cs_val_{}/image'.format(i), im, collections=['cs_val'])
            #tf.summary.image('cs_val_{}/depth'.format(i), depth, collections=['cs_val'])
            #tf.summary.image('cs_val_{}/gt_depth'.format(i), gt_depth, collections=['cs_val'])

        _summarize_value(cs_avgs['allAp'], 'Ap', 'allAp', 'cs_val')
        _summarize_value(cs_avgs['allAp50%'], 'Ap50', 'allAp', 'cs_val')

        for k, v in cs_avgs["classes"].items():
            _summarize_value(v['ap'], k, 'ap', 'cs_val')
            _summarize_value(v['ap50%'], k, 'ap50', 'cs_val')
        for i, k in enumerate(net._losses.keys()):
            _summarize_value(avg_losses[i], k, 'losses', 'cs_val')

        summary_op = tf.summary.merge_all(key='cs_val')
        sess.run(tf.global_variables_initializer())
        summary = sess.run(summary_op)
        writer.add_summary(summary, epoch)

        writer.close()
        print('Done evaluating.')


def _summarize_value(value, name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.Variable(value, dtype=tf.float32, name=name, trainable=False)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p
