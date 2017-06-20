import glob

from datasets.reader import read
from datasets.preprocessing import preprocess_example

# TODO adapt for motion-rcnn after baseline works

def get_example(dataset_name, split_name, records_root,
                batch_size=1, is_training=False):
    file_pattern = dataset_name + '/' + split_name + '/' + '*.tfrecord'
    tfrecords = glob.glob(records_root + '/' + file_pattern)

    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = read(tfrecords)
    image, gt_boxes, gt_masks = preprocess_example(image, gt_boxes, gt_masks, is_training)

    return image, tf.shape(image), gt_boxes, gt_masks
