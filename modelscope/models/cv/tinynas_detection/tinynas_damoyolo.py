# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .detector import SingleStageDetector


@MODELS.register_module(
    Tasks.domain_specific_object_detection,
    module_name=Models.tinynas_damoyolo)
@MODELS.register_module(
    Tasks.image_object_detection, module_name=Models.tinynas_damoyolo)
class DamoYolo(SingleStageDetector):

    def __init__(self, model_dir, *args, **kwargs):
        self.config_name = 'damoyolo.py'
        super(DamoYolo, self).__init__(model_dir, *args, **kwargs)
