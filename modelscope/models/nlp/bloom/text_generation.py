# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import BloomForCausalLM as BloomForCausalLMTransform

from modelscope.metainfo import Models
from modelscope.models import MODELS
from modelscope.utils.constant import Tasks
from .backbone import MsModelMixin, TorchModel


@MODELS.register_module(
    group_key=Tasks.text_generation, module_name=Models.bloom)
class BloomForTextGeneration(MsModelMixin, BloomForCausalLMTransform,
                             TorchModel):

    pass
