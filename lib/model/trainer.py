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

import tensorflow as tf

from model.config import cfg
from utils.timer import Timer
from datasets.cityscapes.evaluate import evaluate_np_preds as evaluate_cs
from layers.instance_image import full_size_mask


class Trainer(object):
    """A wrapper class for the training process."""

    def __init__(self, network_cls, dataset,
                 ckpt_dir, tbdir, pretrained_model=None):
        self.network_cls = network_cls
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

    def train_val(self, schedule):
        for epochs, learning_rate in schedule:
            for epoch in range(epochs):
                self.train_epoch(epoch, learning_rate)
                self.evaluate()

    def train_epoch(self, learning_rate):
        seed = cfg.RNG_INITIAL_SEED + epoch * cfg.RNG_EPOCH_SEED_INCREMENT
        np.random.seed(seed)

        with tf.Graph().as_default():
            tf.set_random_seed(seed)
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            with tf.Session(config=tfconfig) as sess:
                self._train_epoch(sess, learning_rate)

    def _train_epoch(self, sess, learning_rate):
        net = self.network_cls(self.dataset.get_train_batch(),
                               is_training=True,
                               num_classes=self.dataset.num_classes)

        train_op, lr_placeholder = self._get_train_op(net)

        saver = tf.train.Saver(max_to_keep=cfg.CHECKPOINTS_MAX_TO_KEEP,
                               keep_checkpoint_every_n_hours=4)

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt is None:
            epoch = 0
            print('Loading initial model weights from {:s}'.format(self.pretrained_model))
            restorer.restore(sess, self.pretrained_model)
            print('Loaded.')
        else:
            ckpt_path = ckpt.model_checkpoint_path
            epoch = int(ckpt_path.split('/')[-1].split('-')[-1])
            print('Restoring model checkpoint from {:s}'.format(ckpt_path))
            saver.restore(sess, ckpt_path)
            print('Restored.')

        writer = tf.summary.FileWriter(self.tbdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        timer = Timer()
        i = 0
        max_i = cfg.TRAIN.EXAMPLES_PER_EPOCH - 1
        last_summary_time = time.time()
        try:
            while not coord.should_stop():
                timer.tic()
                feed_dict = {lr_placeholder: learning_rate}

                run_ops = [
                    net._losses['rpn_cross_entropy'],
                    net._losses['rpn_loss_box'],
                    net._losses['cross_entropy'],
                    net._losses['loss_box'],
                    net._losses['mask_loss']
                    net._losses['total_loss'],
                    train_op
                ]

                if i % cfg.TRAIN.SUMMARY_INTERVAL == 0:
                    run_ops.append(net._summary_op)

                run_results = sess.run(run_ops, feed_dict=feed_dict)

                if i % cfg.TRAIN.SUMMARY_INTERVAL == 0:
                    summary = run_results[-1]
                    run_results = run_results[:-2]
                    writer.add_summary(summary, float(i - 1) + epoch * max_iter)

                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, mask_loss, total_loss = run_results

                timer.toc()

                if i % cfg.TRAIN.DISPLAY_INTERVAL == 0:
                    print('epoch {} [%d / %d at {:.3f} s/batch] '
                          'loss: {:.4f} ({:.4f}|{:.4f} rpn cls|box, {:.4f}|{:.4f} cls|box, {:.4f} mask)'
                          ' - lr {}'
                          .format(epoch, i, max_i, timer.average_time
                                  total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, mask_loss,
                                  lr.eval()))
                i += 1
        except tf.errors.OutOfRangeError:
            pass

        save_filename = os.path.join(self.ckpt_dir, 'model.ckpt')
        saver.save(sess, save_filename, global_step=epoch)
        writer.close()
        coord.request_stop()
        coord.join(threads)
        print('Finished epoch {} and wrote checkpoint.'.format(epoch))

    def _get_train_op(net):
        loss = net._losses['total_loss']

        lr_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        tf.summary.scalar('TRAIN/learning_rate', lr_placeholder)
        tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

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
                self._evalute(sess)

    def _evaluate(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        assert ckpt is not None

        writer = tf.summary.FileWriter(self.tbvaldir)

        print('Loading model checkpoint for evaluation: {:s}'.format(ckpt_path))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loaded.')

        batch = self.dataset.get_val_batch()
        image = batch[0]
        net = self.network_cls(batch,
                               is_training=False,
                               num_classes=self.dataset.num_classes)

        iters = 0
        avg_losses = np.zeros([len(net._losses)])
        pred_np_arrays = []
        try:
            while not coord.should_stop():
                loss_ops = [v for (k, v) in net._losses]
                pred_ops = [
                    net._predictions['masks'],
                    net._predictions['mask_cls_scores'],
                    net._predictions['mask_scores'],
                    net._predictions['mask_rois']
                ]

                run_results = sess.run(loss_ops + pred_ops + [tf.shape(image)])
                loss_results = run_results[:len(loss_ops)]
                pred_results = run_results[len(loss_ops):]
                image_shape_np = run_results[-1]
                avg_losses += loss_results
                pred_np_arrays.append(pred_results)
                iters += 1
        except tf.errors.OutOfRangeError:
            pass

        avg_losses /= iters # TODO add summaries
        height, width = image_shape_np[1:3]

        pred_lists = []
        for masks, cls_scores, scores, rois in pred_np_arrays:
            for i in masks.shape[0]:
                train_id = np.argmax(cls_scores[i])
                if train_id == 0:
                    # Ignore background class
                    continue
                pred_dct = {}
                pred_dct['imgId'] = "todo"
                pred_dct['labelID'] = labels.trainId2label[train_id].id
                pred_dct['conf'] = scores[i]
                mask = full_size_mask(rois[i, 1:5], masks[i, :, :], height, width)
                pred_dct['binaryMask'] = mask.astype(np.uint8)

        cs_avgs = evaluate_cs(pred_lists) # TODO add summaries

        # TODO add KITTI 2015 eval

        writer.close()
        print('Done evaluating.')
