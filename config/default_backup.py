# Modified based on the MDEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = '/home/limy/codes/SDNet/logs'
_C.GPUS = (0,)
_C.VIZ = False
_C.VIZ_INPUTNORM = True  # only
_C.VIZ_TRAINSET = True  # visualize train set
_C.WORKERS = 2
_C.PRINT_FREQ = 100

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'ddictnet'
_C.MODEL.ARCH = 'wrn'
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.IMAGE_SIZE = [32, 32]  # width * height, ex: 192 * 256

_C.MODEL.NUM_LAYERS = 2
_C.MODEL.ISFISTA = True
_C.MODEL.EXPANSION_FACTOR = 1
_C.MODEL.LAMBDA = [0.0]
_C.MODEL.ADAPTIVELAMBDA = False
_C.MODEL.NONEGATIVE = False # True originally
_C.MODEL.WNORM = True
_C.MODEL.DICTLOSS = False
_C.MODEL.RCLOSS_FACTOR = 0.0
_C.MODEL.ORTHO_COEFF = 0.0
_C.MODEL.MU = 0.0
_C.MODEL.SHORTCUT = True
_C.MODEL.PAD_MODE = 'constant'
_C.MODEL.POOLING = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/limy/codes/SDNet/data/cifar10/'
_C.DATASET.DATASET = 'cifar10'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.NOISE_LEVEL = -1

# train
_C.TRAIN = CN()
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [80, 160]
_C.TRAIN.LR = 0.1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.RESUME = False
_C.TRAIN.MODEL_FILE = ''
_C.TRAIN.BATCH_SIZE_PER_GPU = 128

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 128
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    # if args.testModel:
    #     cfg.TEST.MODEL_FILE = args.testModel

    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
