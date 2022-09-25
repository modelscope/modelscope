# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.ponet import \
    PoNetForMaskedLM as PoNetForMaskedLMTransformer
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['PoNetForMaskedLM']


@MODELS.register_module(Tasks.fill_mask, module_name=Models.ponet)
class PoNetForMaskedLM(TorchModel, PoNetForMaskedLMTransformer):
    """PoNet for MLM model.'.

    Inherited from ponet.PoNetForMaskedLM and TorchModel, so this class can be registered into Model sets.
    """

    def __init__(self, config, model_dir):
        super(TorchModel, self).__init__(model_dir)
        PoNetForMaskedLMTransformer.__init__(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        output = PoNetForMaskedLMTransformer.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            segment_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels)
        output[OutputKeys.INPUT_IDS] = input_ids
        return output

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        return super(PoNetForMaskedLMTransformer,
                     PoNetForMaskedLM).from_pretrained(
                         pretrained_model_name_or_path=model_dir,
                         model_dir=model_dir)
