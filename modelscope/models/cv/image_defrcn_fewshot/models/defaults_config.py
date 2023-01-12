# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/config/defaults.py

from detectron2.config.defaults import _C

_CC = _C

# ----------- Backbone ----------- #
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_AT = 3

# ------------- RPN -------------- #
_CC.MODEL.RPN.FREEZE = False
_CC.MODEL.RPN.ENABLE_DECOUPLE = False
_CC.MODEL.RPN.BACKWARD_SCALE = 1.0

# ------------- ROI -------------- #
_CC.MODEL.ROI_HEADS.NAME = 'Res5ROIHeads'
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
_CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = 'FastRCNNOutputLayers'
_CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
_CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.8
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster

# ------------- TEST ------------- #
_CC.TEST.PCB_ENABLE = False
_CC.TEST.PCB_MODELTYPE = 'resnet'  # res-like
_CC.TEST.PCB_MODELPATH = ''
_CC.TEST.PCB_ALPHA = 0.50
_CC.TEST.PCB_UPPER = 1.0
_CC.TEST.PCB_LOWER = 0.05

# ------------ Other ------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5
_CC.MUTE_HEADER = True
