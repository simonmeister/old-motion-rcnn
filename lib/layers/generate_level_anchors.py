# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister, based on code by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np

from layers.generate_anchors import generate_anchors


def generate_level_anchors(height, width, feat_stride,
                           anchor_scales=[8], anchor_ratios=[0.5, 1, 2]):
    """Generates all anchors over the pyramid level specified by feat_stride.

    We generate A = len(anchor_scales) * len(anchor_ratios) anchors per feature map
    position (K = height * width positions) at that level.

    Args:
        height, width: height and width of the rpn level
        feat_stride: feature stride at the rpn level for which to generate anchors
        anchor_scales: anchor scales to generate relative to feat_stride

    Returns:
        anchors: array of shape (A * K, 4), [[x1, y1, x2, y2], ...]
    """
    anchors = generate_anchors(base_size=feat_stride,
                               ratios=np.array(anchor_ratios),
                               scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()))
    shifts = shifts.transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    return anchors


if __name__ == '__main__':
    a = generate_level_anchors(2, 2, 2, anchor_scales=[2], anchor_ratios=[1])
    print(a)
