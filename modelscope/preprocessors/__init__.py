# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .base import Preprocessor
    from .builder import PREPROCESSORS, build_preprocessor
    from .common import Compose
    from .asr import WavToScp
    from .audio import LinearAECAndFbank
    from .image import (LoadImage, load_image,
                        ImageColorEnhanceFinetunePreprocessor,
                        ImageInstanceSegmentationPreprocessor,
                        ImageDenoisePreprocessor)
    from .kws import WavToLists
    from .multi_modal import (OfaImageCaptionPreprocessor,
                              MPlugVisualQuestionAnsweringPreprocessor)
    from .nlp import (Tokenize, SequenceClassificationPreprocessor,
                      TextGenerationPreprocessor,
                      TokenClassificationPreprocessor, NLIPreprocessor,
                      SentimentClassificationPreprocessor,
                      SentenceSimilarityPreprocessor, FillMaskPreprocessor,
                      ZeroShotClassificationPreprocessor, NERPreprocessor)
    from .space import (DialogIntentPredictionPreprocessor,
                        DialogModelingPreprocessor,
                        DialogStateTrackingPreprocessor)
    from .video import ReadVideoData

else:
    _import_structure = {
        'base': ['Preprocessor'],
        'builder': ['PREPROCESSORS', 'build_preprocessor'],
        'common': ['Compose'],
        'audio': ['LinearAECAndFbank'],
        'asr': ['WavToScp'],
        'video': ['ReadVideoData'],
        'image': [
            'LoadImage', 'load_image', 'ImageColorEnhanceFinetunePreprocessor',
            'ImageInstanceSegmentationPreprocessor', 'ImageDenoisePreprocessor'
        ],
        'kws': ['WavToLists'],
        'multi_modal': [
            'OfaImageCaptionPreprocessor',
            'MPlugVisualQuestionAnsweringPreprocessor'
        ],
        'nlp': [
            'Tokenize', 'SequenceClassificationPreprocessor',
            'TextGenerationPreprocessor', 'TokenClassificationPreprocessor',
            'NLIPreprocessor', 'SentimentClassificationPreprocessor',
            'SentenceSimilarityPreprocessor', 'FillMaskPreprocessor',
            'ZeroShotClassificationPreprocessor', 'NERPreprocessor'
        ],
        'space': [
            'DialogIntentPredictionPreprocessor', 'DialogModelingPreprocessor',
            'DialogStateTrackingPreprocessor', 'InputFeatures'
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
