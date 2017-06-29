# Motion R-CNN

This repository contains the official TensorFlow implementation of
[Motion R-CNN](TODO).

## Requirements

- [tensorflow (>= 1.2.0)](https://www.tensorflow.org/install/install_linux)
- `pip install tensorflow-gpu pillow matplotlib opencv-python easydict cython tqdm`

## Setup
- create `./out` directory
- copy `env_template/env.yml` to `out/env.yml` and adapt for your machine setup
- download `http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz` and unzip to `./data/models/`
- go to `./libs` and run `make`
- run `tools/create_tfrecords.py` with each `--dataset`/`--split` combination you need

## Usage
- run `python test/cityscapes.py` to visualize the cityscapes dataset
- run `python tools/trainval.py` for training
- run `python tools/test.py` for testing

## Acknowledgments
- The code in `lib/nms` and `lib/boxes` is taken without changes from
  [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).
- The tensorflow code in `lib/nets/resnet_v1.py` and `lib/nets/network.py` is based on
  [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn/tree/master/lib/nets).
- The initial files in `lib/layers` are based on
  [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn/tree/master/lib/rpn)
  and include small modifications from
  [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn/tree/master/lib/layer_utils).
- The code in `lib/datasets/cityscapes/cityscapesscripts` is adapted from
  [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).  
- A few functions are loosely inspired by
  [FastMaskRCNN](https://github.com/CharlesShang/FastMaskRCNN).

## License
See [LICENSE](https://github.com/simonmeister/motion-rcnn/blob/master/LICENSE) for details.
