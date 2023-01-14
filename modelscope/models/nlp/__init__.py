# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .bart import BartForTextErrorCorrection
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
    from .csanmt import CsanmtForTranslation
    from .deberta_v2 import DebertaV2ForMaskedLM, DebertaV2Model
    from .gpt_neo import GPTNeoModel
    from .gpt2 import GPT2Model
    from .gpt3 import GPT3ForTextGeneration, DistributedGPT3
    from .gpt_moe import GPTMoEForTextGeneration, DistributedGPTMoE
    from .heads import SequenceClassificationHead
    from .megatron_bert import (
        MegatronBertConfig,
        MegatronBertForMaskedLM,
        MegatronBertModel,
    )
    from .palm_v2 import PalmForTextGeneration
    from .ponet import PoNetForMaskedLM, PoNetModel, PoNetConfig
    from .space import SpaceForDialogIntent, SpaceForDialogModeling, SpaceForDST
    from .space_T_cn import TableQuestionAnswering
    from .space_T_en import StarForTextToSql
    from .structbert import (
        SbertForFaqQuestionAnswering,
        SbertForMaskedLM,
        SbertForSequenceClassification,
        SbertForTokenClassification,
        SbertModel,
    )
    from .T5 import T5ForConditionalGeneration
    from .mglm import MGLMForTextSummarization
    from .codegeex import CodeGeeXForCodeTranslation, CodeGeeXForCodeGeneration
    from .task_models import (
        FeatureExtractionModel,
        InformationExtractionModel,
        LSTMCRFForNamedEntityRecognition,
        LSTMCRFForWordSegmentation,
        LSTMCRFForPartOfSpeech,
        SequenceClassificationModel,
        SingleBackboneTaskModelBase,
        TaskModelForTextGeneration,
        TokenClassificationModel,
        TransformerCRFForNamedEntityRecognition,
        TransformerCRFForWordSegmentation,
    )
    from .veco import (VecoConfig, VecoForMaskedLM,
                       VecoForSequenceClassification,
                       VecoForTokenClassification, VecoModel)
    from .bloom import BloomModel
    from .unite import UniTEModel
    from .use import UserSatisfactionEstimation
else:
    _import_structure = {
        'backbones': ['SbertModel'],
        'bart': ['BartForTextErrorCorrection'],
        'csanmt': ['CsanmtForTranslation'],
        'heads': ['SequenceClassificationHead'],
        'gpt2': ['GPT2Model'],
        'gpt3': ['GPT3ForTextGeneration', 'DistributedGPT3'],
        'gpt_moe': ['GPTMoEForTextGeneration', 'DistributedGPTMoE'],
        'structbert': [
            'SbertForFaqQuestionAnswering',
            'SbertForMaskedLM',
            'SbertForSequenceClassification',
            'SbertForTokenClassification',
            'SbertModel',
        ],
        'veco': [
            'VecoConfig',
            'VecoForMaskedLM',
            'VecoForSequenceClassification',
            'VecoForTokenClassification',
            'VecoModel',
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
        'megatron_bert': [
            'MegatronBertConfig',
            'MegatronBertForMaskedLM',
            'MegatronBertModel',
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
            'LSTMCRFForWordSegmentation',
            'LSTMCRFForPartOfSpeech',
            'SequenceClassificationModel',
            'SingleBackboneTaskModelBase',
            'TaskModelForTextGeneration',
            'TokenClassificationModel',
            'TransformerCRFForNamedEntityRecognition',
            'TransformerCRFForWordSegmentation',
        ],
        'sentence_embedding': ['SentenceEmbedding'],
        'T5': ['T5ForConditionalGeneration'],
        'mglm': ['MGLMForTextSummarization'],
        'codegeex':
        ['CodeGeeXForCodeTranslation', 'CodeGeeXForCodeGeneration'],
        'gpt_neo': ['GPTNeoModel'],
        'bloom': ['BloomModel'],
        'unite': ['UniTEModel'],
        'use': ['UserSatisfactionEstimation']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
