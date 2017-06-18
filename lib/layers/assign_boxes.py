# --------------------------------------------------------
# Tensorflow Mask R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Charles Shang and Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np


def assign_boxes(boxes, min_k=2, max_k=5):
    """Assigns boxes to pyramid levels.

    Args:
        boxes: array of shape (N, 4), [[x1, y1, x2, y2], ...]
        min_k: minimum level index
        max_k: maximum level index
    Returns:
        level_ids: array of shape (N,), per-box level indices
    """
    k0 = 4
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    areas = ws * hs
    k = np.floor(k0 + np.log2(np.sqrt(areas) / 224))
    k = k.astype(np.int32)
    inds = np.where(k < min_k)[0]
    k[inds] = min_k
    inds = np.where(k > max_k)[0]
    k[inds] = max_k
    return k

# TODO write assign boxes in tf??
