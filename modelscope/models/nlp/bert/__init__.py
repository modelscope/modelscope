# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .modeling_bert import (
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

    from .configuration_bert import BertConfig, BertOnnxConfig

else:
    _import_structure = {
        'configuration_bert': ['BertConfig', 'BertOnnxConfig'],
    }

    _import_structure['modeling_bert'] = [
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
