# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.nlp import ModelForTextRanking
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks

logger = logging.get_logger()


@MODELS.register_module(Tasks.text_ranking, module_name=Models.bert)
class BertForTextRanking(ModelForTextRanking):
    r"""Bert Model transformer with a sequence classification/regression head on top
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.

    This model inherits from :class:`SequenceClassificationModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    """
    base_model_type = 'bert'
