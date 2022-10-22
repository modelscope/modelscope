# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .text_error_correction import TextErrorCorrectionPreprocessor
    from .nlp_base import (
        DocumentSegmentationPreprocessor,
        FaqQuestionAnsweringPreprocessor,
        FillMaskPoNetPreprocessor,
        NLPPreprocessor,
        NLPTokenizerPreprocessorBase,
        PassageRankingPreprocessor,
        RelationExtractionPreprocessor,
        SentenceEmbeddingPreprocessor,
        SequenceClassificationPreprocessor,
        TokenClassificationPreprocessor,
        TextGenerationPreprocessor,
        Text2TextGenerationPreprocessor,
        Tokenize,
        WordSegmentationBlankSetToLabelPreprocessor,
        ZeroShotClassificationPreprocessor,
    )
    from .mglm_summarization_preprocessor import mglmSummarizationPreprocessor

else:
    _import_structure = {
        'nlp_base': [
            'DocumentSegmentationPreprocessor',
            'FaqQuestionAnsweringPreprocessor',
            'FillMaskPoNetPreprocessor',
            'NLPPreprocessor',
            'NLPTokenizerPreprocessorBase',
            'PassageRankingPreprocessor',
            'RelationExtractionPreprocessor',
            'SentenceEmbeddingPreprocessor',
            'SequenceClassificationPreprocessor',
            'TokenClassificationPreprocessor',
            'TextGenerationPreprocessor',
            'Tokenize',
            'Text2TextGenerationPreprocessor',
            'WordSegmentationBlankSetToLabelPreprocessor',
            'ZeroShotClassificationPreprocessor',
        ],
        'text_error_correction': [
            'TextErrorCorrectionPreprocessor',
        ],
        'mglm_summarization_preprocessor': ['mglmSummarizationPreprocessor']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
