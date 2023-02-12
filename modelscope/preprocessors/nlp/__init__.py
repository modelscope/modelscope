# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .text_error_correction import TextErrorCorrectionPreprocessor
    from .text_generation_preprocessor import TextGenerationJiebaPreprocessor
    from .bert_seq_cls_tokenizer import Tokenize
    from .document_segmentation_preprocessor import DocumentSegmentationTransformersPreprocessor
    from .faq_question_answering_preprocessor import FaqQuestionAnsweringTransformersPreprocessor
    from .fill_mask_preprocessor import FillMaskPoNetPreprocessor, FillMaskTransformersPreprocessor
    from .text_ranking_preprocessor import TextRankingTransformersPreprocessor
    from .relation_extraction_preprocessor import RelationExtractionTransformersPreprocessor
    from .text_classification_preprocessor import TextClassificationTransformersPreprocessor
    from .sentence_embedding_preprocessor import SentenceEmbeddingTransformersPreprocessor
    from .text_generation_preprocessor import TextGenerationTransformersPreprocessor, \
        TextGenerationT5Preprocessor, TextGenerationSentencePiecePreprocessor, SentencePiecePreprocessor
    from .token_classification_preprocessor import TokenClassificationTransformersPreprocessor, \
        WordSegmentationBlankSetToLabelPreprocessor
    from .token_classification_thai_preprocessor import WordSegmentationPreprocessorThai, NERPreprocessorThai
    from .token_classification_viet_preprocessor import NERPreprocessorViet
    from .zero_shot_classification_preprocessor import ZeroShotClassificationTransformersPreprocessor
    from .space import (DialogIntentPredictionPreprocessor,
                        DialogModelingPreprocessor,
                        DialogStateTrackingPreprocessor, InputFeatures,
                        MultiWOZBPETextField, IntentBPETextField)
    from .space_T_en import ConversationalTextToSqlPreprocessor
    from .space_T_cn import TableQuestionAnsweringPreprocessor
    from .mglm_summarization_preprocessor import MGLMSummarizationPreprocessor
    from .translation_evaluation_preprocessor import TranslationEvaluationPreprocessor
    from .dialog_classification_use_preprocessor import DialogueClassificationUsePreprocessor
    from .document_grounded_dialog_generate_preprocessor import DocumentGroundedDialogGeneratePreprocessor
    from .document_grounded_dialog_retrieval_preprocessor import DocumentGroundedDialogRetrievalPreprocessor
    from .document_grounded_dialog_retrieval_preprocessor import DocumentGroundedDialogRerankPreprocessor
else:
    _import_structure = {
        'bert_seq_cls_tokenizer': ['Tokenize'],
        'document_segmentation_preprocessor':
        ['DocumentSegmentationTransformersPreprocessor'],
        'faq_question_answering_preprocessor':
        ['FaqQuestionAnsweringTransformersPreprocessor'],
        'fill_mask_preprocessor':
        ['FillMaskPoNetPreprocessor', 'FillMaskTransformersPreprocessor'],
        'text_ranking_preprocessor': ['TextRankingTransformersPreprocessor'],
        'relation_extraction_preprocessor':
        ['RelationExtractionTransformersPreprocessor'],
        'text_classification_preprocessor':
        ['TextClassificationTransformersPreprocessor'],
        'sentence_embedding_preprocessor':
        ['SentenceEmbeddingTransformersPreprocessor'],
        'text_generation_preprocessor': [
            'TextGenerationTransformersPreprocessor',
            'TextGenerationJiebaPreprocessor',
            'TextGenerationT5Preprocessor',
            'TextGenerationSentencePiecePreprocessor',
            'SentencePiecePreprocessor',
        ],
        'token_classification_preprocessor': [
            'TokenClassificationTransformersPreprocessor',
            'WordSegmentationBlankSetToLabelPreprocessor'
        ],
        'zero_shot_classification_preprocessor':
        ['ZeroShotClassificationTransformersPreprocessor'],
        'text_error_correction': [
            'TextErrorCorrectionPreprocessor',
        ],
        'mglm_summarization_preprocessor': ['MGLMSummarizationPreprocessor'],
        'token_classification_thai_preprocessor': [
            'NERPreprocessorThai',
            'WordSegmentationPreprocessorThai',
        ],
        'token_classification_viet_preprocessor': [
            'NERPreprocessorViet',
        ],
        'space': [
            'DialogIntentPredictionPreprocessor',
            'DialogModelingPreprocessor',
            'DialogStateTrackingPreprocessor',
            'InputFeatures',
            'MultiWOZBPETextField',
            'IntentBPETextField',
        ],
        'space_T_en': ['ConversationalTextToSqlPreprocessor'],
        'space_T_cn': ['TableQuestionAnsweringPreprocessor'],
        'translation_evaluation_preprocessor':
        ['TranslationEvaluationPreprocessor'],
        'dialog_classification_use_preprocessor':
        ['DialogueClassificationUsePreprocessor'],
        'document_grounded_dialog_generate_preprocessor':
        ['DocumentGroundedDialogGeneratePreprocessor'],
        'document_grounded_dialog_retrieval_preprocessor':
        ['DocumentGroundedDialogRetrievalPreprocessor'],
        'document_grounded_dialog_rerank_preprocessor':
        ['DocumentGroundedDialogRerankPreprocessor']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
