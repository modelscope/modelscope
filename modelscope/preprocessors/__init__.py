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
    from .nlp import (Tokenize, SequenceClassificationPreprocessor,
                      TextGenerationPreprocessor,
                      TokenClassificationPreprocessor,
                      SingleSentenceClassificationPreprocessor,
                      PairSentenceClassificationPreprocessor,
                      FillMaskPreprocessor, ZeroShotClassificationPreprocessor,
                      NERPreprocessor, TextErrorCorrectionPreprocessor,
                      FaqQuestionAnsweringPreprocessor)
    from .slp import DocumentSegmentationPreprocessor
    from .space import (DialogIntentPredictionPreprocessor,
                        DialogModelingPreprocessor,
                        DialogStateTrackingPreprocessor)
    from .video import ReadVideoData
    from .star import ConversationalTextToSqlPreprocessor

else:
    _import_structure = {
        'base': ['Preprocessor'],
        'builder': ['PREPROCESSORS', 'build_preprocessor'],
        'common': ['Compose', 'ToTensor', 'Filter'],
        'audio': ['LinearAECAndFbank'],
        'asr': ['WavToScp'],
        'video': ['ReadVideoData'],
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
            'TextErrorCorrectionPreprocessor',
            'FaqQuestionAnsweringPreprocessor'
        ],
        'slp': ['DocumentSegmentationPreprocessor'],
        'space': [
            'DialogIntentPredictionPreprocessor', 'DialogModelingPreprocessor',
            'DialogStateTrackingPreprocessor', 'InputFeatures'
        ],
        'star': ['ConversationalTextToSqlPreprocessor'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
