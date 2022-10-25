# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import BloomConfig
from transformers import BloomModel as BloomModelTransform

from modelscope.metainfo import Models
from modelscope.models.builder import BACKBONES
from modelscope.utils.constant import Fields


@BACKBONES.register_module(group_key=Fields.nlp, module_name=Models.bloom)
class BloomModel(BloomModelTransform):

    def __init__(self, **kwargs):
        config = BloomConfig(**kwargs)
        super().__init__(config)
