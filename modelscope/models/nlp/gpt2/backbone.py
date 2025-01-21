# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import GPT2Config
from transformers import GPT2Model as GPT2ModelTransform

from modelscope.metainfo import Models
from modelscope.models.builder import BACKBONES
from modelscope.utils.constant import Tasks


@BACKBONES.register_module(group_key=Tasks.backbone, module_name=Models.gpt2)
class GPT2Model(GPT2ModelTransform):

    def __init__(self, **kwargs):
        config = GPT2Config(**kwargs)
        super().__init__(config)
