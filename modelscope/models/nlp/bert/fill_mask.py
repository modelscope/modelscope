# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Heads, Models
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.fill_mask import ModelForFillMask
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks

logger = logging.get_logger()


@MODELS.register_module(Tasks.fill_mask, module_name=Models.bert)
class BertForMaskedLM(ModelForFillMask):

    base_model_type = Models.bert
    head_type = Heads.bert_mlm
