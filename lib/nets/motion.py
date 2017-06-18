# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from nets.utils import crop_rois


# TODO adapt for batches of images!

# takes filtered boxes/masks + results
def flow_from_motions(boxes, masks, depth, motions):
    """

    Args:

    Returns:
    """
    # create global coord grids over whole image TODO
    positions = ...
    _pixels_to_3d(positions, depth, )

    # TODO we first have to expand predicted masks to full image resolution
    # TODO

    # iterate over boxes, select region to transform
    for i in range(boxes.shape[0]):
        pass  # Transform points with box


def _compose_rotations(sin_alpha, sin_beta, sin_gamma):
    """Compose 3d rotations from angle sines.

    Args:
      sin_{alpha, beta, gamma}: tensor of shape (N, 1) with values in [-1, 1]

    Returns:
      rotations: tensor of shape (N, 3, 3)
    """
    zero = tf.zeros_like(sin_alpha)
    one = tf.ones_like(sin_alpha)

    cos_alpha = tf.sqrt(1 - sin_alpha ** 2)
    cos_beta = tf.sqrt(1 - sin_beta ** 2)
    cos_gamma = tf.sqrt(1 - sin_gamma ** 2)

    rot_x_1 = tf.stack([cos_alpha, -sin_alpha, zero], axis=2)
    rot_x_2 = tf.stack([sin_alpha, cos_alpha, zero], axis=2)
    rot_x_3 = tf.stack([zero, zero, one], axis=2)
    rot_x = tf.concat([rot_x_1, rot_x_2, rot_x_3], axis=1)

    rot_y_1 = tf.stack([cos_beta, zero, sin_beta], axis=2)
    rot_y_2 = tf.stack([zero, one, zero], axis=2)
    rot_y_3 = tf.stack([-sin_beta, zero, cos_beta], axis=2)
    rot_y = tf.concat([rot_y_1, rot_y_2, rot_y_3], axis=1)

    rot_z_1 = tf.stack([one, zero, zero], axis=2)
    rot_z_2 = tf.stack([zero, cos_gamma, -sin_gamma], axis=2)
    rot_z_3 = tf.stack([zero, sin_gamma, cos_gamma], axis=2)
    rot_z = tf.concat([rot_z_1, rot_z_2, rot_z_3], axis=1)

    rotations = tf.matmul(rot_x, tf.matmul(rot_y, rot_z))
    return rotations


def _apply_object_motions(points, motions, masks):
    """Transform points with per-object motions, weighted by per-pixel object masks.

    Args:
      points: tensor of shape (h, w, N, 3)
      motions: tensor of shape (N, 9)
      masks: tensor of shape (h, w, N)

    returns:
      points_t: tensor of same shape as 'points'
    """
    pivots = motions[:, 0:3]
    translations = motions[:, 3:6]
    sin_angles = tf.split(motions[:, 6:9], 3, axis=1)
    rotations = _compose_rotations(*sin_angles)

    # broadcast pivot point translation of shape (boxes, 3)
    points_centered = points - pivots

    # rotate the centered points with the rotation matrix of each mask,
    # (boxes, 3, 3), (h, w, boxes, 3) -> (h, w, boxes, 3)
    points_rot_all = tf.einsum('nij,hwnj->hwn', rotations, points_centered)

    # broadcast translation of shape (boxes, 3)
    points_t_all = points_rot_all + translations + pivots

    # compute difference between points and transformed points to obtain increments
    # which we can apply to the original points, weighted by the mask
    diffs = points_t_all - points
    points_t = points + (masks * diffs)

    return points_t


def _apply_camera_motion(points, motion):
    """Transform all points with global camera motion.

    Args:
      points: tensor of shape (h, w, N, 3)
      motion: tensor of shape (9)

    returns:
      points_t: tensor of same shape as 'points'
    """
    pivot = motion[0:3]
    translation = motion[3:6]
    sin_angles = tf.split(tf.expand_dims(motion[6:9], axis=0), 3, axis=1)
    rotation = _compose_rotations(*sin_angles)[0, :]

    # broadcast pivot point translation of shape (3)
    points_centered = points - pivot

    # rotate the centered points with the camera rotation matrix
    # (3, 3), (h, w, boxes, 3) -> (h, w, boxes, 3)
    points_rot = tf.einsum('ij,hwnj->hwn', rotations, points_centered)

    # broadcast translation of shape (3)
    points_t = points_rot + translation + pivot

    return points_t


def _sample_crops_bilinear(image, flow, positions):
    """Sample fixed sized crops from an image.
    For each crop, sampling is done at flow targets for the respective object.

    Args:
      positions: tensor of shape (N, h, w, 2)
      flow: tensor of shape (N, h, w, 2)
      image: tensor of shape (H, W, 3)

    Returns:
      crops: tensor of shape (N, h, w, 3)
    """
    num_boxes, mask_height, mask_width, _ = tf.unstack(tf.shape(positions))
    # num_batch, height, width, channels = tf.unstack(tf.shape(image)) #TODO batch processing
    height, width, channels = tf.unstack(tf.shape(image))

    max_x = tf.cast(width - 1, 'int32')
    max_y = tf.cast(height - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    im_flat = tf.reshape(image, [-1, channels])
    flow_flat = tf.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    # The fractional part is used to control the bilinear interpolation.
    flow_floor = tf.to_int32(tf.floor(flow_flat))
    bilinear_weights = flow_flat - tf.floor(flow_flat)

    # Construct base indices which are displaced with the flow
    pos_x = tf.reshape(positions[:, 0], [-1])
    pos_y = tf.reshape(positions[:, 1], [-1])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]

    # Compute interpolation weights for 4 adjacent pixels
    # expand to num_batch * height * width x 1 for broadcasting in add_n below
    wa = tf.expand_dims((1 - xw) * (1 - yw), 1)  # top left pixel
    wb = tf.expand_dims((1 - xw) * yw, 1)  # bottom left pixel
    wc = tf.expand_dims(xw * (1 - yw), 1)  # top right pixel
    wd = tf.expand_dims(xw * yw, 1)  # bottom right pixel

    x0 = pos_x + x
    x1 = x0 + 1
    y0 = pos_y + y
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # TODO add again for batches to work
    # dim1 = width * height
    # batch_offsets = tf.range(num_batch) * dim1
    # base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
    # base = tf.reshape(base_grid, [-1])

    base_y0 = y0 * width  # + base
    base_y1 = y1 * width  # + base
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    warped = tf.reshape(warped_flat, [num_boxes, mask_height, mask_width, channels])
    return warped


def _pixels_to_3d(positions, d, camera_intrinsics):
    """Map to 3d metric coordinates from 2d pixel coordinates and metric depth.
    """
    x, y = tf.unstack(positions, axis=3)
    f = camera_intrinsics['f']
    x0 = camera_intrinsics['x0']
    y0 = camera_intrinsics['y0']

    X = (x - x0) * d
    Y = (y - y0) * d
    Z = d

    return tf.stack([X, Y, Z], axis=3)


def _3d_to_pixels(points, camera_intrinsics):  # TODO move dim handling (un/stacking) out to make this np/tf generic
    """Project 3d coordinates to 2d pixel coordinates.
    """
    f = camera_intrinsics['f']
    x0 = camera_intrinsics['x0']
    y0 = camera_intrinsics['y0']

    X, Y, Z = tf.unstack(points, axis=3)

    x = f * X / Z + x0
    y = f * Y / Z + y0
    return tf.stack([x, y], axis=3)


def motion_loss(boxes, masks, camera_intrinsics, motions, camera_motion, depth_rois,
                flow_rois=None, images=None):
    """Penalize motion of N objects.

    Args:
      boxes: tensor of shape (N, 4)
      masks: tensor of shape (N, h, w)
      camera_intrinsics: dictionary with keys 'f', 'x0' and 'y0', measured in pixels
      camera_motion: tensor of shape (9)
      depth: tensor of shape (N, H, W, 1).
      flow_crops: None or tensor of shape (N, h, w, 2)
      images: None or two-tuple of tensors, each with shape (B, H, W, 3)

    Returns:
      loss: scalar
    """
    assert flow_crops is not None or images is not None

    # create x and y coord grids for all masks from boxes on CPU (numpy or cython)
    # coordinates refer to full resolution image!
    # (num_boxes, 28, 28)
    # x = ... # TODO (uses boxes)
    # y = ...
    # positions = tf.stack([x, y], axis=3)
    positions = ...

    crop_batch_inds = tf.zeros(tf.shape(boxes)[0])
    mask_height, mask_width = tf.shape(mask)[1:3]

    depth_crops = crop(depth, boxes, crop_batch_inds,
                       pooled_height=mask_height,
                       pooled_width=mask_width)

    # point cloud of shape (boxes, h, w, 3)
    points = _pixels_to_3d(positions, depth_crops, camera_intrinsics)

    # (boxes, h, w, ...) -> (h, w, boxes, ...)
    points = tf.transpose(points, perm=[1, 2, 0, 3])
    masks = tf.transpose(masks, perm=[1, 2, 0])

    points = _apply_object_motions(points, motions, masks)
    points = _apply_camera_motion(points, camera_motion)

    points = tf.transpose(points, perm=[2, 0, 1])

    positions_t = _3d_to_pixels(points, camera_intrinsics)
    flow = positions_t - positions

    if flow_crops is not None:
        pass  # TODO
        # flow - flow_crops
    else:
        image, next_image = images
        sampled_crops = _sample_crops_bilinear(next_image, flow, positions)
        image_crops = crop(image, boxes, crop_batch_inds,
                           pooled_height=mask_height,
                           pooled_width=mask_width)
        # sampled_crops - image_crops


def camera_motion_loss():
    pass


def depth_loss():
    pass
