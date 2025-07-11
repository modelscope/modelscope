# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .bart import BartForTextErrorCorrection
    from .bert import (BertConfig, BertForDocumentSegmentation,
                       BertForMaskedLM, BertForSentenceEmbedding,
                       BertForSequenceClassification, BertForTextRanking,
                       BertForTokenClassification, BertModel, SiameseUieModel)
    from .bloom import BloomForTextGeneration, BloomModel
    from .canmt import CanmtForTranslation
    from .chatglm import (ChatGLMConfig, ChatGLMForConditionalGeneration,
                          ChatGLMTokenizer)
    from .chatglm2 import (ChatGLM2Config, ChatGLM2ForConditionalGeneration,
                           ChatGLM2Tokenizer)
    from .codegeex import CodeGeeXForCodeGeneration, CodeGeeXForCodeTranslation
    from .csanmt import CsanmtForTranslation
    from .deberta_v2 import DebertaV2ForMaskedLM, DebertaV2Model
    from .dgds import (DocumentGroundedDialogGenerateModel,
                       DocumentGroundedDialogRerankModel,
                       DocumentGroundedDialogRetrievalModel)
    from .glm_130b import GLM130bForTextGeneration
    from .gpt2 import GPT2Model
    from .gpt3 import DistributedGPT3, GPT3ForTextGeneration
    from .gpt_moe import DistributedGPTMoE, GPTMoEForTextGeneration
    from .gpt_neo import GPTNeoModel
    from .heads import TextClassificationHead
    from .hf_transformers import TransformersModel
    from .llama import (LlamaConfig, LlamaForTextGeneration, LlamaModel,
                        LlamaTokenizer, LlamaTokenizerFast)
    from .llama2 import (Llama2Config, Llama2ForTextGeneration, Llama2Model,
                         Llama2Tokenizer, Llama2TokenizerFast)
    from .lstm import LSTMForTokenClassificationWithCRF, LSTMModel
    from .megatron_bert import (MegatronBertConfig, MegatronBertForMaskedLM,
                                MegatronBertModel)
    from .mglm import MGLMForTextSummarization
    from .palm_v2 import PalmForTextGeneration
    from .plug_mental import (PlugMentalConfig,
                              PlugMentalForSequenceClassification,
                              PlugMentalModel)
    from .polylm import PolyLMForTextGeneration
    from .ponet import PoNetConfig, PoNetForMaskedLM, PoNetModel
    from .qwen import (QWenConfig, QWenForTextGeneration, QWenModel,
                       QWenTokenizer)
    from .space import (SpaceForDialogIntent, SpaceForDialogModeling,
                        SpaceForDST)
    from .space_T_cn import TableQuestionAnswering
    from .space_T_en import StarForTextToSql
    from .structbert import (SbertForFaqQuestionAnswering, SbertForMaskedLM,
                             SbertForSequenceClassification,
                             SbertForTokenClassification, SbertModel)
    from .T5 import T5ForConditionalGeneration
    from .task_models import (ModelForFeatureExtraction,
                              ModelForInformationExtraction,
                              ModelForMachineReadingComprehension,
                              ModelForTextClassification,
                              ModelForTextGeneration, ModelForTextRanking,
                              ModelForTokenClassification,
                              ModelForTokenClassificationWithCRF,
                              SingleBackboneTaskModelBase)
    from .unite import UniTEForTranslationEvaluation
    from .use import UserSatisfactionEstimation
    from .veco import (VecoConfig, VecoForMaskedLM,
                       VecoForSequenceClassification,
                       VecoForTokenClassification, VecoModel)
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
