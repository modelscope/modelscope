from typing import Any, Dict, Optional, Union

import numpy as np
from transformers import BertForMaskedLM as BertForMaskedLMTransformer

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.structbert import SbertForMaskedLM
from modelscope.models.nlp.veco import \
    VecoForMaskedLM as VecoForMaskedLMTransformer
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['BertForMaskedLM', 'StructBertForMaskedLM', 'VecoForMaskedLM']


@MODELS.register_module(Tasks.fill_mask, module_name=Models.structbert)
class StructBertForMaskedLM(TorchModel, SbertForMaskedLM):

    def __init__(self, config, model_dir):
        super(TorchModel, self).__init__(model_dir)
        SbertForMaskedLM.__init__(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        output = SbertForMaskedLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels)
        output[OutputKeys.INPUT_IDS] = input_ids
        return output

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        return super(SbertForMaskedLM, StructBertForMaskedLM).from_pretrained(
            pretrained_model_name_or_path=model_dir, model_dir=model_dir)


@MODELS.register_module(Tasks.fill_mask, module_name=Models.bert)
class BertForMaskedLM(TorchModel, BertForMaskedLMTransformer):

    def __init__(self, config, model_dir):
        super(TorchModel, self).__init__(model_dir)
        BertForMaskedLMTransformer.__init__(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        output = BertForMaskedLMTransformer.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels)
        output[OutputKeys.INPUT_IDS] = input_ids
        return output

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        return super(BertForMaskedLMTransformer,
                     BertForMaskedLM).from_pretrained(
                         pretrained_model_name_or_path=model_dir,
                         model_dir=model_dir)


@MODELS.register_module(Tasks.fill_mask, module_name=Models.veco)
class VecoForMaskedLM(TorchModel, VecoForMaskedLMTransformer):

    def __init__(self, config, model_dir):
        super(TorchModel, self).__init__(model_dir)
        VecoForMaskedLMTransformer.__init__(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        output = VecoForMaskedLMTransformer.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels)
        output[OutputKeys.INPUT_IDS] = input_ids
        return output

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        return super(VecoForMaskedLMTransformer,
                     VecoForMaskedLM).from_pretrained(
                         pretrained_model_name_or_path=model_dir,
                         model_dir=model_dir)
