# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.detection.detectors import Detection as _Detection

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.image_object_detection, module_name=Models.dino)
class DINO(EasyCVBaseModel, _Detection):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        _Detection.__init__(self, *args, **kwargs)
