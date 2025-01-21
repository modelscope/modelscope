# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Heads, TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.utils.constant import Tasks

__all__ = ['ModelForInformationExtraction']


@MODELS.register_module(
    Tasks.information_extraction,
    module_name=TaskModels.information_extraction)
@MODELS.register_module(
    Tasks.relation_extraction, module_name=TaskModels.information_extraction)
class ModelForInformationExtraction(EncoderModel):
    task = Tasks.information_extraction

    # The default base head type is fill-mask for this head
    head_type = Heads.information_extraction
