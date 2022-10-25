# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .bart import BartForTextErrorCorrection
    from .csanmt import CsanmtForTranslation
    from .heads import SequenceClassificationHead
    from .gpt3 import GPT3ForTextGeneration
    from .palm_v2 import PalmForTextGeneration
    from .space_T_en import StarForTextToSql
    from .space_T_cn import TableQuestionAnswering
    from .space import SpaceForDialogIntent, SpaceForDialogModeling, SpaceForDST
    from .ponet import PoNetForMaskedLM, PoNetModel, PoNetConfig
    from .structbert import (
        SbertForFaqQuestionAnswering,
        SbertForMaskedLM,
        SbertForSequenceClassification,
        SbertForTokenClassification,
        SbertTokenizer,
        SbertTokenizerFast,
    )
    from .bert import (
        BertForMaskedLM,
        BertForTextRanking,
        BertForSentenceEmbedding,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertForDocumentSegmentation,
        BertModel,
        BertConfig,
    )
    from .veco import VecoModel, VecoConfig, VecoForTokenClassification, \
        VecoForSequenceClassification, VecoForMaskedLM, VecoTokenizer, VecoTokenizerFast
    from .deberta_v2 import DebertaV2ForMaskedLM, DebertaV2Model
    from .task_models import (
        FeatureExtractionModel,
        InformationExtractionModel,
        LSTMCRFForNamedEntityRecognition,
        SequenceClassificationModel,
        SingleBackboneTaskModelBase,
        TaskModelForTextGeneration,
        TokenClassificationModel,
        TransformerCRFForNamedEntityRecognition,
    )

    from .T5 import T5ForConditionalGeneration
    from .gpt_neo import GPTNeoModel
else:
    _import_structure = {
        'backbones': ['SbertModel'],
        'bart': ['BartForTextErrorCorrection'],
        'csanmt': ['CsanmtForTranslation'],
        'heads': ['SequenceClassificationHead'],
        'gpt3': ['GPT3ForTextGeneration'],
        'structbert': [
            'SbertForFaqQuestionAnswering',
            'SbertForMaskedLM',
            'SbertForSequenceClassification',
            'SbertForTokenClassification',
            'SbertTokenizer',
            'SbertTokenizerFast',
        ],
        'veco': [
            'VecoModel', 'VecoConfig', 'VecoForTokenClassification',
            'VecoForSequenceClassification', 'VecoForMaskedLM',
            'VecoTokenizer', 'VecoTokenizerFast'
        ],
        'bert': [
            'BertForMaskedLM',
            'BertForTextRanking',
            'BertForSentenceEmbedding',
            'BertForSequenceClassification',
            'BertForTokenClassification',
            'BertForDocumentSegmentation',
            'BertModel',
            'BertConfig',
        ],
        'ponet': ['PoNetForMaskedLM', 'PoNetModel', 'PoNetConfig'],
        'palm_v2': ['PalmForTextGeneration'],
        'deberta_v2': ['DebertaV2ForMaskedLM', 'DebertaV2Model'],
        'space_T_en': ['StarForTextToSql'],
        'space_T_cn': ['TableQuestionAnswering'],
        'space':
        ['SpaceForDialogIntent', 'SpaceForDialogModeling', 'SpaceForDST'],
        'task_models': [
            'FeatureExtractionModel',
            'InformationExtractionModel',
            'LSTMCRFForNamedEntityRecognition',
            'SequenceClassificationModel',
            'SingleBackboneTaskModelBase',
            'TaskModelForTextGeneration',
            'TokenClassificationModel',
            'TransformerCRFForNamedEntityRecognition',
        ],
        'sentence_embedding': ['SentenceEmbedding'],
        'T5': ['T5ForConditionalGeneration'],
        'gpt_neo': ['GPTNeoModel'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
