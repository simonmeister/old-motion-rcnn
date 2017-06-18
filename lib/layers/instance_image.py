# --------------------------------------------------------
# Tensorflow Mask R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np
import cv2

from model.config import cfg
from libs.boxes.bbox_transform import clip_boxes


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
        h = rois[i, 3] - rois[i, 1] + 1
        w = rois[i, 2] - rois[i, 0] + 1
        x = rois[i, 0]
        y = rois[i, 1]
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        m *= i
        mask[y:(y + h), x:(x + w)] = m

    return mask
