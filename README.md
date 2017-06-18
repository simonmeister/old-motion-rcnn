# Motion R-CNN

This repository contains the official TensorFlow implementation of
[Motion R-CNN](TODO).

## Requirements

- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [Resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

## How-to
1. Go to `./libs/datasets/pycocotools` and run `make`
2. run `python download_and_convert_data.py`
3. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz`, unzip it, place it into `./data/pretrained_models/`
4. Go to `./libs` and run `make`
5. run `python train/train.py` for training
6. There are certainly some bugs, please report them back, and let's solve them together.

## Acknowledgment
This repository uses code from
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
- [FastMaskRCNN](https://github.com/CharlesShang/FastMaskRCNN)

## License
See [LICENSE](https://github.com/simonmeister/motion-rcnn/blob/master/LICENSE) for details.
