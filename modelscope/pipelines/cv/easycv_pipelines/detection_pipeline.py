# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Pipelines
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from .base import EasyCVPipeline


@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.easycv_detection)
class EasyCVDetectionPipeline(EasyCVPipeline):
    """Pipeline for easycv detection task."""

    def __init__(self, model: str, model_file_pattern='*.pt', *args, **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """

        super(EasyCVDetectionPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)
