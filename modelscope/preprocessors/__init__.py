# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .base import Preprocessor
    from .builder import PREPROCESSORS, build_preprocessor
    from .common import Compose, ToTensor, Filter
    from .asr import WavToScp
    from .audio import LinearAECAndFbank
    from .image import (LoadImage, load_image,
                        ImageColorEnhanceFinetunePreprocessor,
                        ImageInstanceSegmentationPreprocessor,
                        ImageDenoisePreprocessor)
    from .kws import WavToLists
    from .multi_modal import (OfaPreprocessor, MPlugPreprocessor)
    from .nlp import (
        Tokenize, SequenceClassificationPreprocessor,
        TextGenerationPreprocessor, TokenClassificationPreprocessor,
        SingleSentenceClassificationPreprocessor,
        PairSentenceClassificationPreprocessor, FillMaskPreprocessor,
        ZeroShotClassificationPreprocessor, NERPreprocessor,
        TextErrorCorrectionPreprocessor, FaqQuestionAnsweringPreprocessor,
        SequenceLabelingPreprocessor, RelationExtractionPreprocessor,
        DocumentSegmentationPreprocessor, FillMaskPoNetPreprocessor,
        PassageRankingPreprocessor, SentenceEmbeddingPreprocessor,
        Text2TextGenerationPreprocessor,
        WordSegmentationBlankSetToLabelPreprocessor)
    from .space import (DialogIntentPredictionPreprocessor,
                        DialogModelingPreprocessor,
                        DialogStateTrackingPreprocessor)
    from .video import ReadVideoData, MovieSceneSegmentationPreprocessor
    from .star import ConversationalTextToSqlPreprocessor
    from .star3 import TableQuestionAnsweringPreprocessor

else:
    _import_structure = {
        'base': ['Preprocessor'],
        'builder': ['PREPROCESSORS', 'build_preprocessor'],
        'common': ['Compose', 'ToTensor', 'Filter'],
        'audio': ['LinearAECAndFbank'],
        'asr': ['WavToScp'],
        'video': ['ReadVideoData', 'MovieSceneSegmentationPreprocessor'],
        'image': [
            'LoadImage', 'load_image', 'ImageColorEnhanceFinetunePreprocessor',
            'ImageInstanceSegmentationPreprocessor', 'ImageDenoisePreprocessor'
        ],
        'kws': ['WavToLists'],
        'multi_modal': ['OfaPreprocessor', 'MPlugPreprocessor'],
        'nlp': [
            'Tokenize', 'SequenceClassificationPreprocessor',
            'TextGenerationPreprocessor', 'TokenClassificationPreprocessor',
            'SingleSentenceClassificationPreprocessor',
            'PairSentenceClassificationPreprocessor', 'FillMaskPreprocessor',
            'ZeroShotClassificationPreprocessor', 'NERPreprocessor',
            'SentenceEmbeddingPreprocessor', 'PassageRankingPreprocessor',
            'TextErrorCorrectionPreprocessor',
            'FaqQuestionAnsweringPreprocessor', 'SequenceLabelingPreprocessor',
            'RelationExtractionPreprocessor',
            'Text2TextGenerationPreprocessor',
            'WordSegmentationBlankSetToLabelPreprocessor',
            'DocumentSegmentationPreprocessor', 'FillMaskPoNetPreprocessor'
        ],
        'space': [
            'DialogIntentPredictionPreprocessor', 'DialogModelingPreprocessor',
            'DialogStateTrackingPreprocessor', 'InputFeatures'
        ],
        'star': ['ConversationalTextToSqlPreprocessor'],
        'star3': ['TableQuestionAnsweringPreprocessor'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
