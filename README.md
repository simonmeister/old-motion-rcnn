# Motion R-CNN

This repository contains the official TensorFlow implementation of
[Motion R-CNN](TODO).

## Requirements

- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [Resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

## Setup
- create `./output` directory
- copy `env_template/env.yml` to `output/env.yml` and adapt for your machine setup
- clone `https://github.com/mcordts/cityscapesScripts` to `./data/cityscapes/cityscapesScripts`
- download `http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz` and unzip to `./data/models/`
- go to `./libs` and run `make`
- go to `./data/cityscapes/cityscapesScripts` and run `python setup.py build_ext --inplace`
- run `create_tfrecords.py` with each `--dataset`/`--split` combination you need

## Usage
- run `python tools/trainval.py` for training

## Acknowledgments
This repository uses code from
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
- [FastMaskRCNN](https://github.com/CharlesShang/FastMaskRCNN)

## License
See [LICENSE](https://github.com/simonmeister/motion-rcnn/blob/master/LICENSE) for details.
