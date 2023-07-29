# --------------------------------------------------------
# We adopt this configuration method like swin-transfomer
# (https://github.com/microsoft/Swin-Transformer.git),
# made publicly available under the MIT License
# --------------------------------------------------------'

import os

import yaml
from yacs.config import CfgNode

cfg = CfgNode()

# Base config files
cfg.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
cfg.DATA = CfgNode()
# Batch size for a single GPU, could be overwritten by command line argument
# Number of data loading threads
cfg.DATA.NUM_WORKERS = 8
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
cfg.DATA.PIN_MEMORY = True
# some dataset supported
cfg.DATA.DATASET_NAME = 'Matterport3D'  # Stanford3D, PanoSUNCG3D, 3D60, Pano3D
cfg.DATA.IMG_WIDTH = 1024
cfg.DATA.IMG_HEIGHT = 512

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
cfg.MODEL = CfgNode()
cfg.MODEL.DECODER_DIM = 256
cfg.MODEL.USE_CAF_FUSION = True
cfg.MODEL.MAX_DEPTH = 10.0

# -----------------------------------------------------------------------------
# Backbone settings
# -----------------------------------------------------------------------------
cfg.BACKBONE = CfgNode()
cfg.BACKBONE.TYPE = 'swin'

# for swin
cfg.BACKBONE.NAME = 'swin_base_patch4_window7_224'
cfg.BACKBONE.VERSION = 'base'
cfg.BACKBONE.PRETRAIN_RES = 224
cfg.BACKBONE.PRETRAIN_IMAGENET = '22k'
cfg.BACKBONE.DROP_PATH_RATE = 0.5
cfg.BACKBONE.EMBED_DIM = 128
cfg.BACKBONE.DEPTHS = [2, 2, 18, 2]
cfg.BACKBONE.NUM_HEADS = [4, 8, 16, 32]
cfg.BACKBONE.WINDOW_SIZE = 7
cfg.BACKBONE.FROZEN_STAGES = -1
cfg.BACKBONE.PRETRAIN_LR_SCALE = 1.0

# for resnet
cfg.BACKBONE.RESNET_LAYER_NUM = 50

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
cfg.TRAIN = CfgNode()
# device
cfg.TRAIN.RANK = 0
cfg.TRAIN.WORLD_SIZE = 1
cfg.TRAIN.BACKEND = 'nccl'
cfg.TRAIN.IP_ADDR = 'localhost:'
cfg.TRAIN.PORT = '8000'
cfg.TRAIN.INIT_METHOD = 'tcp://' + cfg.TRAIN.IP_ADDR + cfg.TRAIN.PORT
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.EPOCHS = 300
cfg.TRAIN.WARMUP_EPOCHS = 5
cfg.TRAIN.WEIGHT_DECAY = 0.05
cfg.TRAIN.BASE_LR = 0.001
cfg.TRAIN.WARMUP_LR = 0.001
cfg.TRAIN.MIN_LR = 0.001
# Clip gradient norm
cfg.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
cfg.TRAIN.AUTO_RESUME = True
cfg.TRAIN.RESUME = ''
# Gradient accumulation steps
cfg.TRAIN.ACCUMULATION_STEPS = 0
# LR scheduler
cfg.TRAIN.LR_SCHEDULER = CfgNode()
cfg.TRAIN.LR_SCHEDULER.NAME = 'step'
# Epoch interval to decay LR, used in StepLRScheduler
cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 3
# LR decay rate, used in StepLRScheduler
cfg.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.9
# Optimizer
cfg.TRAIN.OPTIMIZER = CfgNode()
cfg.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
cfg.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer BetasWEIGHT_DECAY
cfg.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
cfg.TRAIN.OPTIMIZER.MOMENTUM = 0.9
cfg.TRAIN.PRETRAINED_MODEL = ''
cfg.TRAIN.LOSS_TYPE = 'RMSLE'  # or 'BerHu'

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
# trained on depth estimation
cfg.AUG = CfgNode()
cfg.AUG.BRIGHTNESS = 0.2
cfg.AUG.CONTRAST = 0.2
cfg.AUG.SATURATION = 0.2
cfg.AUG.HUE = 0.1

# -----------------------------------------------------------------------------
# Testing or validating settings
# -----------------------------------------------------------------------------
cfg.TESTING = CfgNode()
cfg.TESTING.BATCH_SIZE = 1
cfg.TESTING.RANK = 0
cfg.TESTING.WORLD_SIZE = 1
cfg.TESTING.BACKEND = 'nccl'
cfg.TESTING.IP_ADDR = 'localhost:'
cfg.TESTING.PORT = '8000'
cfg.TESTING.INIT_METHOD = 'tcp://' + cfg.TESTING.IP_ADDR + cfg.TESTING.PORT
cfg.TESTING.MEDIAN_ALIGN = False
# -----------------------------------------------------------------------------
# Misc settings
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
cfg.AMP_OPT_LEVEL = 'O0'
# Frequency to save checkpoint
cfg.SAVE_FREQ = 1
# Frequency to logging info
cfg.PRINT_FREQ = 5
# Fixed random seed
cfg.SEED = 0
cfg.OUT_ROOT_DIR = './train'
cfg.EVAL_MODE = False


def update_cfg_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            update_cfg_from_file(config,
                                 os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def get_config(cfg_file):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = cfg.clone()
    update_cfg_from_file(config, cfg_file)

    return config
