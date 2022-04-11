# Modified based on the MDEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = '/home/limy/codes/SDNet-CTRL/saved_models'

# common params for NETWORK
_C.MODEL = CN()

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

# train
_C.TRAIN = CN()
_C.TRAIN.DATASET = 'CIFAR10' #CIFAR10 or MNIST
_C.TRAIN.LR = 0.0002
_C.TRAIN.LR_SCHE_FACTOR = 0.1
_C.TRAIN.LR_SCHE_STEP = [100, 200]
_C.TRAIN.OPTIMIZER = 'Adam' 
_C.TRAIN.ADAM_BETA1 = 0.0
_C.TRAIN.ADAM_BETA2 = 0.9
_C.TRAIN.EPOCHS = 500
_C.TRAIN.BATCH_SIZE = 4096
_C.TRAIN.ARCH = 'INVERSE2'
_C.TRAIN.NZ = 512
_C.TRAIN.NGF = 64
_C.TRAIN.N_ITER_DIS = 1
_C.TRAIN.GAM1 = 1.0
_C.TRAIN.GAM2 = 1.0
_C.TRAIN.EPS = 0.5
_C.TRAIN.MODE = 'multi'

# ## train
# _C.TRAIN = CN()
# _C.TRAIN.DATASET = 'MNIST' #CIFAR10 or MNIST
# _C.TRAIN.LR = 0.0001
# _C.TRAIN.LR_SCHE_FACTOR = 0.1
# _C.TRAIN.LR_SCHE_STEP = [100, 200]
# _C.TRAIN.OPTIMIZER = 'Adam' 
# _C.TRAIN.ADAM_BETA1 = 0.0
# _C.TRAIN.ADAM_BETA2 = 0.9
# _C.TRAIN.EPOCHS = 200
# _C.TRAIN.BATCH_SIZE = 2048
# _C.TRAIN.ARCH = 'INVERSE2'
# _C.TRAIN.NZ = 128
# _C.TRAIN.NGF = 64
# _C.TRAIN.N_ITER_DIS = 1
# _C.TRAIN.GAM1 = 1.0
# _C.TRAIN.GAM2 = 1.0
# _C.TRAIN.EPS = 0.5
# _C.TRAIN.MODE = 'binary'

# testing
# _C.TEST = CN()
# _C.TEST.BATCH_SIZE_PER_GPU = 128
# _C.TEST.MODEL_FILE = ''


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
