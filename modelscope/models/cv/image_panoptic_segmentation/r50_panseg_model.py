# Copyright (c) Alibaba, Inc. and its affiliates.

from easycv.models.segmentation import Mask2Former

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.image_segmentation,
    module_name=Models.r50_panoptic_segmentation)
class R50PanopticSegmentation(EasyCVBaseModel, Mask2Former):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        Mask2Former.__init__(self, *args, **kwargs)
