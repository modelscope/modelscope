# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import BloomConfig
from transformers import BloomForCausalLM as BloomForCausalLMTransform

from modelscope.metainfo import Models
from modelscope.models import MODELS
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    group_key=Tasks.text_generation, module_name=Models.bloom)
class BloomForTextGeneration(BloomForCausalLMTransform):

    def __init__(self, **kwargs):
        config = BloomConfig(**kwargs)
        super().__init__(config)
