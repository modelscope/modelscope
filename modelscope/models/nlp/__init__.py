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
        SiameseUieModel,
    )
    from .bloom import BloomModel, BloomForTextGeneration
    from .codegeex import CodeGeeXForCodeTranslation, CodeGeeXForCodeGeneration
    from .glm_130b import GLM130bForTextGeneration
    from .csanmt import CsanmtForTranslation
    from .canmt import CanmtForTranslation
    from .polylm import PolyLMForTextGeneration
    from .deberta_v2 import DebertaV2ForMaskedLM, DebertaV2Model
    from .chatglm import ChatGLMForConditionalGeneration, ChatGLMTokenizer, ChatGLMConfig
    from .chatglm2 import ChatGLM2ForConditionalGeneration, ChatGLM2Tokenizer, ChatGLM2Config
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
        ModelForMachineReadingComprehension,
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
    from .llama import LlamaForTextGeneration, LlamaConfig, LlamaModel, LlamaTokenizer, LlamaTokenizerFast
    from .llama2 import Llama2ForTextGeneration, Llama2Config, Llama2Model, Llama2Tokenizer, Llama2TokenizerFast
    from .qwen import QWenForTextGeneration, QWenConfig, QWenModel, QWenTokenizer
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
            'SiameseUieModel',
        ],
        'bloom': ['BloomModel', 'BloomForTextGeneration'],
        'csanmt': ['CsanmtForTranslation'],
        'canmt': ['CanmtForTranslation'],
        'polylm': ['PolyLMForTextGeneration'],
        'codegeex':
        ['CodeGeeXForCodeTranslation', 'CodeGeeXForCodeGeneration'],
        'glm_130b': ['GLM130bForTextGeneration'],
        'deberta_v2': ['DebertaV2ForMaskedLM', 'DebertaV2Model'],
        'chatglm': [
            'ChatGLMForConditionalGeneration', 'ChatGLMTokenizer',
            'ChatGLMConfig'
        ],
        'chatglm2': [
            'ChatGLM2ForConditionalGeneration', 'ChatGLM2Tokenizer',
            'ChatGLM2Config'
        ],
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
        'veco': [
            'VecoConfig',
            'VecoForMaskedLM',
            'VecoForSequenceClassification',
            'VecoForTokenClassification',
            'VecoModel',
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
            'ModelForMachineReadingComprehension',
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
        'xlm_roberta': ['XLMRobertaConfig', 'XLMRobertaModel'],
        'llama': [
            'LlamaForTextGeneration', 'LlamaConfig', 'LlamaModel',
            'LlamaTokenizer', 'LlamaTokenizerFast'
        ],
        'llama2': [
            'Llama2ForTextGeneration', 'Llama2Config', 'Llama2Model',
            'Llama2Tokenizer', 'Llama2TokenizerFast'
        ],
        'qwen':
        ['QWenForTextGeneration', 'QWenConfig', 'QWenModel', 'QWenTokenizer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
