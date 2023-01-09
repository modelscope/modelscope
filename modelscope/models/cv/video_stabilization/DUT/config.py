# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict

__C = edict()
cfg = __C
"""
Model options
"""
__C.MODEL = edict()

# gaussian kernel size
__C.MODEL.GAUSSIAN_KSIZE = 15

# gaussian kernel sigma
__C.MODEL.GAUSSIAN_SIGMA = 0.5

# Descriptor Threshold
__C.MODEL.DES_THRSH = 1.0

# Coordinate Threshold
__C.MODEL.COO_THRSH = 5.0

# Ksize
__C.MODEL.KSIZE = 3

# padding
__C.MODEL.padding = 1

# dilation
__C.MODEL.dilation = 1

# scale_list
__C.MODEL.scale_list = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]

# grid size
__C.MODEL.PIXELS = 16

# neighbor radius
__C.MODEL.RADIUS = 200

# input height
__C.MODEL.HEIGHT = 480

# input width
__C.MODEL.WIDTH = 640

# normalization amplitude in optical flow
__C.MODEL.FLOWC = 20

# cluster threshold
__C.MODEL.THRESHOLDPOINT = 102
"""
Training options
"""
__C.TRAIN = edict()

# score strength weight
__C.TRAIN.score_com_strength = 100.0

# scale strength weight
__C.TRAIN.scale_com_strength = 100.0

# non maximum supression threshold
__C.TRAIN.NMS_THRESH = 0.0

# nms kernel size
__C.TRAIN.NMS_KSIZE = 5

# top k patch
__C.TRAIN.TOPK = 512
"""
Threshold options
"""
__C.Threshold = edict()

__C.Threshold.MANG = 2
__C.Threshold.ROT = 5
"""
Infer options
"""
__C.INFER = edict()
__C.INFER.ALLINFER = False
