# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .detector import SingleStageDetector


@MODELS.register_module(
    Tasks.image_object_detection, module_name=Models.tinynas_detection)
class TinynasDetector(SingleStageDetector):

    def __init__(self, model_dir, *args, **kwargs):

        super(TinynasDetector, self).__init__(model_dir, *args, **kwargs)
