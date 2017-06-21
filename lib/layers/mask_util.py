# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Charles Shang
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np
import cv2

from model.config import cfg
from utils.bbox_transform import clip_boxes
from datasets.cityscapes.labels import trainId2label


def instange_image(rois, mask_preds, height, width):
    """Returns image with per-pixel object indices from mask predictions.

    Args:
        rois: (M, 5)
        mask_preds: (M, 28, 28, 1)
        height: height of full image
        width: width of full image

    Returns:
        masks: (height, width), values in [0, M]
    """
    mask = np.zeros((height, width), dtype=np.float32)
    rois = clip_boxes(rois, (height, width))

    for i in range(rois.shape[0]):
        m = mask_preds[i, :, :, 0]
        h = rois[i, 4] - rois[i, 2] + 1
        w = rois[i, 3] - rois[i, 1] + 1
        x = rois[i, 1]
        y = rois[i, 2]
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        m *= i
        mask[y:(y + h), x:(x + w)] = m

    return mask


def binary_mask(roi, mask, height, width):
    full_size = np.zeros((height, width), dtype=np.float32)
    h = box[4] - box[2] + 1
    w = box[3] - box[1] + 1
    x = box[1]
    y = box[2]
    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask[y:(y + h), x:(x + w)] = 1


def color_mask(rois, classes, mask_preds, height, width):
    """Returns full size mask as color image where the color
    of a pixel represents the object class of a active (== 1) mask at that point.

    Args:
        rois: (M, 5)
        classes: (M,)
        mask_preds: (M, 28, 28, 1)
        height: height of full image
        width: width of full image

    Returns:
        masks: (height, width, 3)
    """
    mask = np.zeros((height, width, 3), dtype=np.float32)
    rois = clip_boxes(rois, (height, width))

    for i in range(rois.shape[0]):
        m = mask_preds[i, :, :, 0]
        color = trainId2label[classes[i]].color

        h = rois[i, 4] - rois[i, 2] + 1
        w = rois[i, 3] - rois[i, 1] + 1
        x = rois[i, 1]
        y = rois[i, 2]
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        m *= color
        mask[y:(y + h), x:(x + w)] = m

    return mask
