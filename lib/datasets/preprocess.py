# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Charles Shang
# --------------------------------------------------------
import tensorflow as tf

from model.config import cfg, get_key


def preprocess_example(image, gt_boxes, gt_masks, is_training=False, normalize=True):
    cfg_key = get_key(is_training)
    scale = cfg[cfg_key].SCALE

    image = tf.cast(image, tf.float32)
    if normalize:
        image = normalize_image(image)

    image, gt_boxes, gt_masks = resize(scale, image, gt_boxes, gt_masks)

    if is_training and cfg[cfg_key].USE_FLIPPED:
        image, gt_boxes, gt_masks = random_flip(image, gt_boxes, gt_masks)

    return image, gt_boxes, gt_masks


def normalize_image(image):
    image = image / 255.0
    image -= (cfg.PIXEL_MEANS / 255.0)
    return image


def _resize_gt_masks(gt_masks, height, width):
    gt_masks = tf.expand_dims(gt_masks, -1)
    gt_masks = tf.image.resize_bilinear(gt_masks, [height, width])
    gt_masks = tf.to_float(gt_masks > 0.5)
    gt_masks = tf.squeeze(gt_masks, axis=[-1])
    return gt_masks


def _resize_image(image, height, width):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width])
    image = tf.squeeze(image, axis=[0])
    return image


def _resize_gt_boxes(gt_boxes, scale):
    coords = gt_boxes[:, 0:4] * scale
    classes = gt_boxes[:, 4:5]
    gt_boxes = tf.concat([coords, classes], axis=1)
    return gt_boxes


def resize(shorter_side, image, gt_boxes, gt_masks):
    """Resizes ground truth example consistently such that the shorter side
    of the image is inside the range given as shorter_side.
    """
    height, width = tf.unstack(tf.shape(image))[:2]
    if isinstance(shorter_side, list):
        shorter_side = tf.random_uniform([], shorter_side[0], shorter_side[1],
                                         dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    shorter_side = tf.to_float(shorter_side)
    scale = tf.cond(tf.greater(height, width),
                    lambda: shorter_side / width,
                    lambda: shorter_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    image = _resize_image(image, new_height, new_width)
    gt_masks = _resize_gt_masks(gt_masks, new_height, new_width)
    gt_boxes = _resize_gt_boxes(gt_boxes, scale)

    return image, gt_boxes, gt_masks


def _flip_gt_boxes(gt_boxes, width):
    width = tf.to_float(width)
    x1, y1, x2, y2, classes = tf.split(gt_boxes, 5, axis=1)
    x1 = width - x1
    x2 = width - x2
    return tf.concat([x2, y1, x1, y2, classes], axis=1)


def _flip_gt_masks(gt_masks):
    return tf.reverse(gt_masks, axis=[2])


def _flip_image(image):
    return tf.reverse(image, axis=[1])


def random_flip(image, gt_boxes, gt_masks):
    """Returns randomly horizontally flipped example."""
    width = tf.shape(image)[1]
    return tf.cond(
        tf.greater_equal(tf.random_uniform([]), 0.5),
        lambda: (_flip_image(image),
                 _flip_gt_boxes(gt_boxes, width),
                 _flip_gt_masks(gt_masks)),
        lambda: (image, gt_boxes, gt_masks))
