# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .text_error_correction import TextErrorCorrectionPreprocessor
    from .nlp_base import (NLPTokenizerPreprocessorBase, NLPBasePreprocessor)
    from .text_generation_jieba_preprocessor import TextGenerationJiebaPreprocessor
    from .sentence_piece_preprocessor import SentencePiecePreprocessor
    from .bert_seq_cls_tokenizer import Tokenize
    from .document_segmentation_preprocessor import DocumentSegmentationPreprocessor
    from .faq_question_answering_preprocessor import FaqQuestionAnsweringPreprocessor
    from .fill_mask_preprocessor import FillMaskPoNetPreprocessor, NLPPreprocessor
    from .text_ranking_preprocessor import TextRankingPreprocessor
    from .relation_extraction_preprocessor import RelationExtractionPreprocessor
    from .sentence_classification_preprocessor import SequenceClassificationPreprocessor
    from .sentence_embedding_preprocessor import SentenceEmbeddingPreprocessor
    from .text_generation_preprocessor import TextGenerationPreprocessor
    from .text2text_generation_preprocessor import Text2TextGenerationPreprocessor
    from .token_classification_preprocessor import TokenClassificationPreprocessor, \
        WordSegmentationBlankSetToLabelPreprocessor
    from .token_classification_thai_preprocessor import WordSegmentationPreprocessorThai, NERPreprocessorThai
    from .token_classification_viet_preprocessor import NERPreprocessorViet
    from .zero_shot_classification_reprocessor import ZeroShotClassificationPreprocessor
    from .space import (DialogIntentPredictionPreprocessor,
                        DialogModelingPreprocessor,
                        DialogStateTrackingPreprocessor, InputFeatures,
                        MultiWOZBPETextField, IntentBPETextField)
    from .space_T_en import ConversationalTextToSqlPreprocessor
    from .space_T_cn import TableQuestionAnsweringPreprocessor
    from .mglm_summarization_preprocessor import MGLMSummarizationPreprocessor
else:
    _import_structure = {
        'nlp_base': [
            'NLPTokenizerPreprocessorBase',
            'NLPBasePreprocessor',
        ],
        'text_generation_jieba_preprocessor':
        ['TextGenerationJiebaPreprocessor'],
        'sentence_piece_preprocessor': ['SentencePiecePreprocessor'],
        'bert_seq_cls_tokenizer': ['Tokenize'],
        'document_segmentation_preprocessor':
        ['DocumentSegmentationPreprocessor'],
        'faq_question_answering_preprocessor':
        ['FaqQuestionAnsweringPreprocessor'],
        'fill_mask_preprocessor':
        ['FillMaskPoNetPreprocessor', 'NLPPreprocessor'],
        'text_ranking_preprocessor': ['TextRankingPreprocessor'],
        'relation_extraction_preprocessor': ['RelationExtractionPreprocessor'],
        'sentence_classification_preprocessor':
        ['SequenceClassificationPreprocessor'],
        'sentence_embedding_preprocessor': ['SentenceEmbeddingPreprocessor'],
        'text_generation_preprocessor': ['TextGenerationPreprocessor'],
        'text2text_generation_preprocessor':
        ['Text2TextGenerationPreprocessor'],
        'token_classification_preprocessor': [
            'TokenClassificationPreprocessor',
            'WordSegmentationBlankSetToLabelPreprocessor'
        ],
        'zero_shot_classification_reprocessor':
        ['ZeroShotClassificationPreprocessor'],
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
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
