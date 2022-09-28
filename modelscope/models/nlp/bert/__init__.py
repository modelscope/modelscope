# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .modeling_bert import (
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertLayer,
        BertLMHeadModel,
        BertModel,
        BertPreTrainedModel,
        load_tf_weights_in_bert,
    )

    from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig, BertOnnxConfig
    from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer
    from .tokenization_bert_fast import BertTokenizerFast

else:
    _import_structure = {
        'configuration_bert':
        ['BERT_PRETRAINED_CONFIG_ARCHIVE_MAP', 'BertConfig', 'BertOnnxConfig'],
        'tokenization_bert':
        ['BasicTokenizer', 'BertTokenizer', 'WordpieceTokenizer'],
    }
    _import_structure['tokenization_bert_fast'] = ['BertTokenizerFast']

    _import_structure['modeling_bert'] = [
        'BERT_PRETRAINED_MODEL_ARCHIVE_LIST',
        'BertForMaskedLM',
        'BertForMultipleChoice',
        'BertForNextSentencePrediction',
        'BertForPreTraining',
        'BertForQuestionAnswering',
        'BertForSequenceClassification',
        'BertForTokenClassification',
        'BertLayer',
        'BertLMHeadModel',
        'BertModel',
        'BertPreTrainedModel',
        'load_tf_weights_in_bert',
    ]

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
