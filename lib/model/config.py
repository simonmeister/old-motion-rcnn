from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import yaml
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from model.config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Training schedule
__C.TRAIN.LEARNING_RATES = [0.001]
__C.TRAIN.EPOCHS = [1]

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.SUMMARY_INTERVAL = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# The maximum number of checkpoints stored, older ones are deleted to save space
__C.TRAIN.CHECKPOINTS_MAX_TO_KEEP = 5

# The iteration interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 10

# Scale to use during training
# The scale is the pixel size of an image's shortest side
# Can be a single number or a tuple with min/max for random scale sampling
__C.TRAIN.SCALE = [800, 1024]

# Number of examples per batch
__C.TRAIN.BATCH_SIZE = 1

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered positive (if >= MASK_THRESH)
__C.TRAIN.MASK_THRESH = 0.5

# Maximum number of ground truth masks to sample per image during training
__C.TRAIN.MASKS_TOP_N = 64

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5


# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = False
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Minimum amount of examples in a shuffle queue, more means better shuffling
__C.TRAIN.MIN_EXAMPLES_AFTER_DEQUEUE = 500

# Number of examples in one epoch.
# For cityscapes, this is the number of examples in the train split.
# TODO this is not correct, as some examples were not created due to having no boxes?
__C.TRAIN.EXAMPLES_PER_EPOCH = 2975

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALE = 1024

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_logits layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

# Maximum number of ground truth masks to process per image during testing
__C.TEST.MASKS_TOP_N = 100

#
# ResNet options
#

__C.RESNET = edict()

# Whether to tune the batch nomalization parameters during training
__C.RESNET.BN_TRAIN = True

#
# MISC
#

# Pixel mean values (RGB order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])

# For reproducibility and consistency after re-starting
__C.RNG_INITIAL_SEED = 0
__C.RNG_EPOCH_SEED_INCREMENT = 10000

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Read environment setup from path that is not version controlled
env_cfg_file = osp.join(__C.ROOT_DIR, 'output', 'env.yml')
assert osp.isfile(env_cfg_file), \
    'copy `env_template/env.yml` to `output/env.yml` and adapt for your machine setup'
with open(env_cfg_file, 'r') as f:
    env_cfg = edict(yaml.load(f))

# See env_template/env.yml
__C.TFRECORD_DIR = osp.abspath(env_cfg.TFRECORD_DIR)
__C.DATA_DIR = osp.abspath(env_cfg.DATA_DIR)
__C.CHECKPOINT_DIR = osp.abspath(env_cfg.CHECKPOINT_DIR)

# Number of examples per tfrecord file
__C.EXAMPLES_PER_TFRECORD = 500

# Where to store experiment output data other than checkpoints
__C.LOG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'outputs', 'logs'))
__C.CONFIG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'outputs', 'cfgs'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5, 1, 2]


def get_key(is_training):
    return 'TRAIN' if is_training else 'TEST'


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def write_cfg_to_file(filename):
    with open(filename, 'w') as f:
        yaml.dump(__C, f)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
