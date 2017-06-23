# Motion R-CNN

This repository contains the official TensorFlow implementation of
[Motion R-CNN](TODO).

## Requirements

- [tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- `pip install tensorflow-gpu pillow opencv-python easydict`


## Setup
- create `./output` directory
- copy `env_template/env.yml` to `output/env.yml` and adapt for your machine setup
- TODO clone `https://github.com/mcordts/cityscapesScripts` to `./data`
- download `http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz` and unzip to `./data/models/`
- go to `./libs` and run `make`
- TODO go to `./data/cityscapesScripts` and run `python setup.py build_ext --inplace`
- run `create_tfrecords.py` with each `--dataset`/`--split` combination you need

## Usage
- run `python test/cityscapes.py` to visualize the cityscapes ground truth
- run `python tools/trainval.py` for training
- run `python tools/test.py` for testing

## Acknowledgments
This repository includes code from
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
- [FastMaskRCNN](https://github.com/CharlesShang/FastMaskRCNN)
- [cityscapesScripts](https://github.com/mcordts/cityscapesScripts)

## License
See [LICENSE](https://github.com/simonmeister/motion-rcnn/blob/master/LICENSE) for details.
