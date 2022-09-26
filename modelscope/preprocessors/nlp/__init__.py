# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .text_error_correction import TextErrorCorrectionPreprocessor
    from .nlp_base import (
        Tokenize, SequenceClassificationPreprocessor,
        TextGenerationPreprocessor, TokenClassificationPreprocessor,
        SingleSentenceClassificationPreprocessor,
        Text2TextGenerationPreprocessor,
        PairSentenceClassificationPreprocessor, FillMaskPreprocessor,
        ZeroShotClassificationPreprocessor, NERPreprocessor,
        FaqQuestionAnsweringPreprocessor, SequenceLabelingPreprocessor,
        RelationExtractionPreprocessor, DocumentSegmentationPreprocessor,
        FillMaskPoNetPreprocessor, PassageRankingPreprocessor,
        SentenceEmbeddingPreprocessor,
        WordSegmentationBlankSetToLabelPreprocessor)

else:
    _import_structure = {
        'nlp_base': [
            'Tokenize', 'SequenceClassificationPreprocessor',
            'TextGenerationPreprocessor', 'TokenClassificationPreprocessor',
            'SingleSentenceClassificationPreprocessor',
            'PairSentenceClassificationPreprocessor', 'FillMaskPreprocessor',
            'ZeroShotClassificationPreprocessor', 'NERPreprocessor',
            'SentenceEmbeddingPreprocessor', 'PassageRankingPreprocessor',
            'FaqQuestionAnsweringPreprocessor', 'SequenceLabelingPreprocessor',
            'RelationExtractionPreprocessor',
            'Text2TextGenerationPreprocessor',
            'WordSegmentationBlankSetToLabelPreprocessor',
            'DocumentSegmentationPreprocessor', 'FillMaskPoNetPreprocessor'
        ],
        'text_error_correction': [
            'TextErrorCorrectionPreprocessor',
        ],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
