# Copyright (c) Alibaba, Inc. and its affiliates.

from transformers import BertForMaskedLM as BertForMaskedLMTransformer

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.deberta_v2 import \
    DebertaV2ForMaskedLM as DebertaV2ForMaskedLMTransformer
from modelscope.models.nlp.structbert import SbertForMaskedLM
from modelscope.models.nlp.veco import \
    VecoForMaskedLM as VecoForMaskedLMTransformer
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['BertForMaskedLM', 'StructBertForMaskedLM', 'VecoForMaskedLM']


@MODELS.register_module(Tasks.fill_mask, module_name=Models.structbert)
class StructBertForMaskedLM(TorchModel, SbertForMaskedLM):
    """Structbert for MLM model.

    Inherited from structbert.SbertForMaskedLM and TorchModel, so this class can be registered into Model sets.
    """

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
    """Bert for MLM model.

    Inherited from transformers.BertForMaskedLM and TorchModel, so this class can be registered into Model sets.
    """

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
    """Veco for MLM model.

    Inherited from veco.VecoForMaskedLM and TorchModel, so this class can be registered into Model sets.
    """

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


@MODELS.register_module(Tasks.fill_mask, module_name=Models.deberta_v2)
class DebertaV2ForMaskedLM(TorchModel, DebertaV2ForMaskedLMTransformer):
    """Deberta v2 for MLM model.

    Inherited from deberta_v2.DebertaV2ForMaskedLM and TorchModel, so this class can be registered into Model sets.
    """

    def __init__(self, config, model_dir):
        super(TorchModel, self).__init__(model_dir)
        DebertaV2ForMaskedLMTransformer.__init__(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        output = DebertaV2ForMaskedLMTransformer.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels)
        output[OutputKeys.INPUT_IDS] = input_ids
        return output

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.get('model_dir')
        return super(DebertaV2ForMaskedLMTransformer,
                     DebertaV2ForMaskedLM).from_pretrained(
                         pretrained_model_name_or_path=model_dir,
                         model_dir=model_dir)
