# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:11   xin      1.0         None
'''

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = 'mfn'
_C.MODEL.BACKBONE = 'resnet50_ibn_a'
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.LABEL_SMOOTH = True
_C.MODEL.PRETRAIN_PATH = 'C:/Users/xin/.torch/models/r50_ibn_a.pth'
_C.MODEL.USE_DROPOUT = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.09661545, 0.18356957, 0.21322473]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.13422933, 0.14724616, 0.19259872]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('competition')
# Root PATH to the dataset
_C.DATASETS.DATA_PATH = r'E:\data\reid\dataset8'
# PATH to train set
_C.DATASETS.TRAIN_PATH = r'train'
# PATH to query set
_C.DATASETS.QUERY_PATH = 'query'
# PATH to gallery set
_C.DATASETS.GALLERY_PATH = 'gallery'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.FP16 = False

_C.SOLVER.MAX_EPOCHS = 300

_C.SOLVER.BASE_LR = 0.03
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

_C.SOLVER.GAMMA = 0.6
_C.SOLVER.STEPS = [60, 120]

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCH = 10
_C.SOLVER.WARMUP_BEGAIN_RL = 3e-4
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 32


# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.WEIGHT = ""
_C.TEST.DEBUG = False
_C.TEST.MULTI_GPU = False
_C.TEST.CMC = [1,5,10]
_C.TEST.VIS = False
_C.TEST.VIS_Q_NUM = 10
_C.TEST.VIS_G_NUM = 5
_C.TEST.RERANK = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "E:/data/reid/output"

# Alias for easy usage
cfg = _C