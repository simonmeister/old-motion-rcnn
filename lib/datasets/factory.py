import glob

from .reader import read


# TODO adapt for motion-rcnn after baseline works
# TODO adapt calls to pass records_root
# TODO adapt preprocess script

def get_dataset(dataset_name, split_name, records_root,
                batch_size=1, is_training=False):
    file_pattern = dataset_name + '/' + split_name + '/' + '*.tfrecord'
    tfrecords = glob.glob(records_root + '/' + file_pattern)

    # TODO return dict instead
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = read(tfrecords)

    # TODO resize!?
    # image, gt_boxes, gt_masks = preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training)

    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id
