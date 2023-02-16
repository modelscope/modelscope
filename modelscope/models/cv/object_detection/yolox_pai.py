# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.detection.detectors import YOLOX as _YOLOX

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.cv.easycv_base import EasyCVBaseModel
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.image_object_detection, module_name=Models.yolox)
@MODELS.register_module(
    group_key=Tasks.image_object_detection,
    module_name=Models.image_object_detection_auto)
@MODELS.register_module(
    group_key=Tasks.domain_specific_object_detection, module_name=Models.yolox)
class YOLOX(EasyCVBaseModel, _YOLOX):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        _YOLOX.__init__(self, *args, **kwargs)
