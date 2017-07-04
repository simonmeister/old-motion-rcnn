# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Charles Shang
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np
import cv2

from model.config import cfg
from boxes.bbox_transform import clip_boxes
from datasets.cityscapes.cityscapesscripts.labels import trainId2label


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


def binary_mask(roi, mask_pred, height, width):
    out = np.zeros((height, width), dtype=np.float32)
    h = int(roi[4] - roi[2] + 1)
    w = int(roi[3] - roi[1] + 1)
    x = int(roi[1])
    y = int(roi[2])
    mask = cv2.resize(mask_pred, (w, h), interpolation=cv2.INTER_LINEAR)
    out[y:(y + h), x:(x + w)] = mask
    return out


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
    boxes = clip_boxes(rois[:, 1:], (height, width))

    for i in range(rois.shape[0]):
        m = mask_preds[i, :, :, 0]
        color = trainId2label[int(classes[i])].color

        h = int(round(boxes[i, 3]) - round(boxes[i, 1]) + 1)
        w = int(round(boxes[i, 2]) - round(boxes[i, 0]) + 1)
        x = int(round(boxes[i, 0]))
        y = int(round(boxes[i, 1]))
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.expand_dims(m, axis=2) * color
        mask[y:(y + h), x:(x + w)] += m

    return mask
