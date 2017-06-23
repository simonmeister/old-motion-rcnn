# --------------------------------------------------------
# Motion R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Simon Meister
# --------------------------------------------------------

import os
import sys
import math
import random
import json
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

import datasets.cityscapes.labels as labels
from model.config import cfg


_DATA_URLS = [
    "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
    "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
]


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _read_raw(paths):
    path_queue = tf.train.string_input_producer(
        paths, shuffle=False, capacity=len(paths), num_epochs=1)
    reader = tf.WholeFileReader()
    _, raw = reader.read(path_queue)
    return raw


def _read_image(paths, dtype, channels=1):
    raw = _read_raw(paths)
    return tf.image.decode_png(raw, channels=channels, dtype=dtype)


def _get_instance_masks_and_boxes(instance_img, mask_height=28, mask_width=28):
    """Get instance level ground truth.

    Note: instance_img is expected to consist of regular ids, not trainIds.

    Returns:
      masks: (m, h, w) numpy array
      boxes: (m, 5), [[x1, y1, x2, y2, class], ...]
    """
    all_ids = np.unique(instance_img).tolist()
    class_ids = [label.id for label in labels.labels]
    pixel_ids_of_instances = [i for i in all_ids if i not in class_ids]

    masks = []
    cropped_masks = []
    boxes = []
    classes = []
    for pixel_id in pixel_ids_of_instances:
        class_id = pixel_id // 1000
        train_id = labels.id2label[class_id].trainId

        mask = instance_img == pixel_id
        nonzero_y, nonzero_x = np.nonzero(np.squeeze(mask))
        y1 = np.min(nonzero_y)
        y2 = np.max(nonzero_y)
        x1 = np.min(nonzero_x)
        x2 = np.max(nonzero_x)

        box = np.array([x1, y1, x2, y2, train_id], dtype=np.float32)
        mask = mask.astype(np.uint8)

        masks.append(mask)
        boxes.append(box)

    if len(boxes) > 0:
        boxes = np.stack(boxes, axis=0)
        masks = np.stack(masks, axis=0)
    else:
        boxes = np.zeros((0, 5))
        masks = np.zeros((0, mask_height, mask_width))
    return masks, boxes


def _get_record_filename(record_dir, shard_id, num_shards):
    output_filename = '{0:05d}-of-{0:05d}.tfrecord'.format(shard_id, num_shards)
    return os.path.join(record_dir, output_filename)


def _collect_files(modality_dir):
    paths = []
    for city_name in sorted(os.listdir(modality_dir)):
        city_dir = os.path.join(modality_dir, city_name)
        for i, filename in enumerate(sorted(os.listdir(city_dir))):
            path = os.path.join(city_dir, filename)
            paths.append(path)
    return paths


def _create_tfexample(img_id, img, next_img, disparity_img, instance_img,
                      camera, vehicle):
    b = camera['extrinsic']['baseline']
    f = (camera['intrinsic']['fx'] + camera['intrinsic']['fy']) / 2.
    disparity = (disparity_img.astype(float) - 1.) / 256.
    depth = b * f / disparity
    depth[disparity_img == 0] = 0
    x0 = camera['intrinsic']['u0']
    y0 = camera['intrinsic']['v0']
    # TODO variable for seqs of variable length... use sequence info to get exacter
    frame_rate = 8.5
    # TODO calc from sequence data for more precise info when skipping frames - e.g. avg.
    # TODO do we have to use the camera extrinsics to get exact translation / yaw?
    yaw = vehicle['yawRate'] / frame_rate
    translation = vehicle['speed'] / frame_rate

    height, width = img.shape[:2]
    masks, boxes = _get_instance_masks_and_boxes(instance_img)
    num_instances = boxes.shape[0]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/id': _bytes_feature(img_id.encode('utf8')),
        'image/encoded': _bytes_feature(img.tostring()),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'next_image/encoded': _bytes_feature(next_img.tostring()),
        'label/num_instances': _int64_feature(num_instances),
        'label/boxes': _bytes_feature(boxes.tostring()),
        'label/masks': _bytes_feature(masks.tostring()),
        'label/depth': _bytes_feature(depth.tostring()),
        'label/camera/intrinsics/f': _float_feature(f),
        'label/camera/intrinsics/x0': _float_feature(x0),
        'label/camera/intrinsics/y0': _float_feature(y0),
        'label/camera/motion/yaw': _float_feature(yaw),
        'label/camera/motion/translation': _float_feature(translation),
    }))
    return example, num_instances


# TODO extract multiple examples at different framerates per annotated file
def _write_tfrecord(record_dir, dataset_dir, split_name, is_training=False):
    """Loads image files and writes files to a TFRecord.
    Note: masks and bboxes will lose shape info after converting to string.
    """
    print('processing data from {}'.format(split_name))

    image_dir = os.path.join(dataset_dir, 'leftImg8bit', split_name)
    sequence_dir = os.path.join(dataset_dir, 'sequence', split_name)
    gt_dir = os.path.join(dataset_dir, 'gtFine', split_name)
    disparity_dir = os.path.join(dataset_dir, 'disparity', split_name)
    camera_dir = os.path.join(dataset_dir, 'camera', split_name)
    vehicle_dir = os.path.join(dataset_dir, 'vehicle', split_name)
    sequence_dir = os.path.join(dataset_dir, 'sequence', split_name)

    image_paths = []
    next_paths = []
    image_ids = []
    for city_name in sorted(os.listdir(image_dir)):
        city_dir = os.path.join(image_dir, city_name)
        files = sorted(os.listdir(city_dir))
        print("collecting {} examples from city {}".format(len(files), city_name))
        for i, filename in enumerate(files):
            path = os.path.join(city_dir, filename)
            image_paths.append(path)
            number_str = filename.split('_')[1].split('_')[0]
            next_file = "{}_{}_000021_leftImg8bit.png".format(city_name, number_str)
            next_path = os.path.join(sequence_dir, city_name, next_file)
            next_paths.append(next_path)
            image_ids.append("{}_{}".format(city_name, i))

    instance_paths = []
    for city_dirname in sorted(os.listdir(gt_dir)):
        city_dir = os.path.join(gt_dir, city_dirname)
        for filename in sorted(os.listdir(city_dir)):
            path = os.path.join(city_dir, filename)
            if path.endswith('instanceIds.png'):
                instance_paths.append(path)

    vehicle_paths = _collect_files(vehicle_dir)
    camera_paths = _collect_files(camera_dir)
    disparity_paths = _collect_files(disparity_dir)

    assert len(image_paths) == len(vehicle_paths) == len(camera_paths) == len(disparity_paths) \
           == len(next_paths) == len(instance_paths)

    if is_training:
        zipped = list(zip(image_paths, image_ids, instance_paths))
        random.seed(0)
        random.shuffle(zipped)
        image_paths, image_ids, instance_paths = zip(*zipped)

    num_per_shard = cfg.EXAMPLES_PER_TFRECORD
    num_shards = int(math.ceil(len(image_ids) / float(num_per_shard)))

    print('creating max. {} examples in {} shards with at most {} examples each'
          .format(len(image_ids), num_shards, num_per_shard))

    created_count = 0

    for shard_id in range(num_shards):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            record_filename = _get_record_filename(record_dir, shard_id, num_shards)
            options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
            with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(image_ids))

                shard_instance_paths = instance_paths[start_ndx:end_ndx]
                shard_image_paths = image_paths[start_ndx:end_ndx]
                shard_next_paths = next_paths[start_ndx:end_ndx]
                shard_disparity_paths = disparity_paths[start_ndx:end_ndx]

                img_ = _read_image(shard_image_paths, dtype=tf.uint8, channels=3)
                next_img_ = _read_image(shard_next_paths, dtype=tf.uint8, channels=3)
                instance_img_ = _read_image(shard_instance_paths, dtype=tf.uint16)
                disparity_img_ = _read_image(shard_disparity_paths, dtype=tf.uint16)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    tf.train.start_queue_runners()

                    for i in range(start_ndx, end_ndx):
                        if i % 1 == 0:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
                                i + 1, len(image_ids), shard_id))
                            sys.stdout.flush()

                        img_id = image_ids[i]
                        img, instance_img, disparity_img, next_img = sess.run(
                            [img_, instance_img_, disparity_img_, next_img_])

                        with open(camera_paths[i]) as camera_file:
                            camera = json.load(camera_file)

                        with open(vehicle_paths[i]) as vehicle_file:
                            vehicle = json.load(vehicle_file)

                        example, num_instances = _create_tfexample(
                            img_id, img, next_img, disparity_img,
                            instance_img, camera, vehicle)

                        if num_instances > 0 or is_training == False:
                            created_count += 1
                            tfrecord_writer.write(example.SerializeToString())
                        else:
                            print("Skipping example {}: 0 instances".format(i))
    print("Created {} examples ({} skipped)."
          .format(created_count, len(image_ids) - create_records))
    sys.stdout.write('\n')
    sys.stdout.flush()


def create_records(records_root, dataset_root, split_name='train'):
    assert split_name in ['train', 'val', 'test', 'mini'], split_name
    is_training = split_name in ['train']

    # if not tf.gfile.Exists(dataset_root):
    #  tf.gfile.MakeDirs(dataset_root)

    # for url in _DATA_URLS:
    #   download_and_uncompress_zip(url, dataset_dir)
    #   TODO automatically create mini split by copying test/bonn

    record_dir = os.path.join(records_root, 'cityscapes', split_name)

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    _write_tfrecord(record_dir,
                    os.path.join(dataset_root, 'cityscapes'),
                    split_name,
                    is_training=is_training)

    print("\nFinished converting cityscapes '{}' split".format(split_name))
