import numpy as np

import _init_paths
from layers.assign_to_levels import assign_to_levels
from layers.generate_level_anchors import generate_level_anchors


im_size = [1024, 2048]
for index, feat_stride in enumerate([64, 32, 16, 8, 4]):
    anchor_boxes = generate_level_anchors(im_size[0] / feat_stride, im_size[1] / feat_stride,
                                          feat_stride=feat_stride)
    indices = assign_to_levels(anchor_boxes[:3, :], im_size, 4)
    print(anchor_boxes[1, 2] - anchor_boxes[1, 1] + 1, indices)
    for i in range(3):
        assert index == indices[i]
