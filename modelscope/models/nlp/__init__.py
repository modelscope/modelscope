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
    from .bloom import BloomModel
    from .codegeex import CodeGeeXForCodeTranslation, CodeGeeXForCodeGeneration
    from .csanmt import CsanmtForTranslation
    from .deberta_v2 import DebertaV2ForMaskedLM, DebertaV2Model
    from .gpt_neo import GPTNeoModel
    from .gpt2 import GPT2Model
    from .gpt3 import GPT3ForTextGeneration, DistributedGPT3
    from .gpt_moe import GPTMoEForTextGeneration, DistributedGPTMoE
    from .heads import TextClassificationHead
    from .hf_transformers import TransformersModel
    from .lstm import (
        LSTMModel,
        LSTMForTokenClassificationWithCRF,
    )
    from .megatron_bert import (
        MegatronBertConfig,
        MegatronBertForMaskedLM,
        MegatronBertModel,
    )
    from .mglm import MGLMForTextSummarization
    from .palm_v2 import PalmForTextGeneration
    from .plug_mental import (PlugMentalConfig, PlugMentalModel,
                              PlugMentalForSequenceClassification)
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
    from .task_models import (
        ModelForFeatureExtraction,
        ModelForInformationExtraction,
        ModelForTextClassification,
        SingleBackboneTaskModelBase,
        ModelForTextGeneration,
        ModelForTextRanking,
        ModelForTokenClassification,
        ModelForTokenClassificationWithCRF,
    )
    from .unite import UniTEForTranslationEvaluation
    from .use import UserSatisfactionEstimation
    from .veco import (VecoConfig, VecoForMaskedLM,
                       VecoForSequenceClassification,
                       VecoForTokenClassification, VecoModel)
    from .dgds import (DocumentGroundedDialogGenerateModel,
                       DocumentGroundedDialogRetrievalModel,
                       DocumentGroundedDialogRerankModel)
    from .xlm_roberta import XLMRobertaConfig, XLMRobertaModel

else:
    _import_structure = {
        'bart': ['BartForTextErrorCorrection'],
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
        'bloom': ['BloomModel'],
        'csanmt': ['CsanmtForTranslation'],
        'codegeex':
        ['CodeGeeXForCodeTranslation', 'CodeGeeXForCodeGeneration'],
        'deberta_v2': ['DebertaV2ForMaskedLM', 'DebertaV2Model'],
        'heads': ['TextClassificationHead'],
        'hf_transformers': ['TransformersModel'],
        'gpt2': ['GPT2Model'],
        'gpt3': ['GPT3ForTextGeneration', 'DistributedGPT3'],
        'gpt_moe': ['GPTMoEForTextGeneration', 'DistributedGPTMoE'],
        'gpt_neo': ['GPTNeoModel'],
        'structbert': [
            'SbertForFaqQuestionAnswering',
            'SbertForMaskedLM',
            'SbertForSequenceClassification',
            'SbertForTokenClassification',
            'SbertModel',
        ],
        'lstm': [
            'LSTM',
            'LSTMForTokenClassificationWithCRF',
        ],
        'megatron_bert': [
            'MegatronBertConfig',
            'MegatronBertForMaskedLM',
            'MegatronBertModel',
        ],
        'mglm': ['MGLMForTextSummarization'],
        'palm_v2': ['PalmForTextGeneration'],
        'plug_mental': [
            'PlugMentalConfig',
            'PlugMentalModel',
            'PlugMentalForSequenceClassification',
        ],
        'ponet': ['PoNetForMaskedLM', 'PoNetModel', 'PoNetConfig'],
        'space_T_en': ['StarForTextToSql'],
        'space_T_cn': ['TableQuestionAnswering'],
        'space':
        ['SpaceForDialogIntent', 'SpaceForDialogModeling', 'SpaceForDST'],
        'task_models': [
            'ModelForFeatureExtraction',
            'ModelForInformationExtraction',
            'ModelForTextClassification',
            'SingleBackboneTaskModelBase',
            'ModelForTextGeneration',
            'ModelForTextRanking',
            'ModelForTokenClassification',
            'ModelForTokenClassificationWithCRF',
        ],
        'sentence_embedding': ['SentenceEmbedding'],
        'T5': ['T5ForConditionalGeneration'],
        'unite': ['UniTEForTranslationEvaluation'],
        'use': ['UserSatisfactionEstimation'],
        'dgds': [
            'DocumentGroundedDialogGenerateModel',
            'DocumentGroundedDialogRetrievalModel',
            'DocumentGroundedDialogRerankModel'
        ],
        'veco': [
            'VecoConfig',
            'VecoForMaskedLM',
            'VecoForSequenceClassification',
            'VecoForTokenClassification',
            'VecoModel',
        ],
        'xlm_roberta': ['XLMRobertaConfig', 'XLMRobertaModel'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
