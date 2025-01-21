# The implementation is adopted from OSTrack,
# made publicly available under the MIT License at https://github.com/botaoye/OSTrack/
from easydict import EasyDict as edict

cfg = edict()

# MODEL
cfg.MODEL = edict()

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = 'vit_base_patch16_224_ce'
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.1
cfg.MODEL.BACKBONE.CE_LOC = [3, 6, 9]
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = [0.7, 0.7, 0.7]
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'CTR_POINT'

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = 'CENTER'
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 384
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 192

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 192
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 384
