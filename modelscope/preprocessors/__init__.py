# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .asr import WavToScp
    from .audio import AudioBrainPreprocessor, LinearAECAndFbank
    from .base import Preprocessor
    from .builder import PREPROCESSORS, build_preprocessor
    from .common import Compose, Filter, ToTensor
    from .cv import (ControllableImageGenerationPreprocessor,
                     ImageClassificationMmcvPreprocessor,
                     ImageRestorationPreprocessor)
    from .image import (ImageColorEnhanceFinetunePreprocessor,
                        ImageDeblurPreprocessor, ImageDenoisePreprocessor,
                        ImageInstanceSegmentationPreprocessor, LoadImage,
                        load_image)
    from .kws import WavToLists
    from .multi_modal import (DiffusionImageGenerationPreprocessor,
                              HiTeAPreprocessor,
                              ImageCaptioningClipInterrogatorPreprocessor,
                              MplugOwlPreprocessor, MPlugPreprocessor,
                              OfaPreprocessor)
    from .nlp import (CanmtTranslationPreprocessor,
                      ConversationalTextToSqlPreprocessor,
                      DialogIntentPredictionPreprocessor,
                      DialogModelingPreprocessor,
                      DialogStateTrackingPreprocessor,
                      DialogueClassificationUsePreprocessor,
                      DocumentGroundedDialogGeneratePreprocessor,
                      DocumentGroundedDialogRerankPreprocessor,
                      DocumentGroundedDialogRetrievalPreprocessor,
                      DocumentSegmentationTransformersPreprocessor,
                      FaqQuestionAnsweringTransformersPreprocessor,
                      FillMaskPoNetPreprocessor,
                      FillMaskTransformersPreprocessor,
                      MachineReadingComprehensionForNERPreprocessor,
                      MGLMSummarizationPreprocessor, NERPreprocessorThai,
                      NERPreprocessorViet,
                      RelationExtractionTransformersPreprocessor,
                      SentenceEmbeddingTransformersPreprocessor,
                      SentencePiecePreprocessor, SiameseUiePreprocessor,
                      TableQuestionAnsweringPreprocessor,
                      TextClassificationTransformersPreprocessor,
                      TextErrorCorrectionPreprocessor,
                      TextGenerationJiebaPreprocessor,
                      TextGenerationSentencePiecePreprocessor,
                      TextGenerationT5Preprocessor,
                      TextGenerationTransformersPreprocessor,
                      TextRankingTransformersPreprocessor,
                      TokenClassificationTransformersPreprocessor, Tokenize,
                      TranslationEvaluationTransformersPreprocessor,
                      WordAlignmentPreprocessor,
                      WordSegmentationBlankSetToLabelPreprocessor,
                      WordSegmentationPreprocessorThai,
                      ZeroShotClassificationTransformersPreprocessor)
    from .tts import KanttsDataPreprocessor
    from .video import MovieSceneSegmentationPreprocessor, ReadVideoData

else:
    _import_structure = {
        'base': ['Preprocessor'],
        'builder': ['PREPROCESSORS', 'build_preprocessor'],
        'common': ['Compose', 'ToTensor', 'Filter'],
        'audio': ['LinearAECAndFbank', 'AudioBrainPreprocessor'],
        'asr': ['WavToScp'],
        'video': ['ReadVideoData', 'MovieSceneSegmentationPreprocessor'],
        'image': [
            'LoadImage', 'load_image', 'ImageColorEnhanceFinetunePreprocessor',
            'ImageInstanceSegmentationPreprocessor',
            'ImageDenoisePreprocessor', 'ImageDeblurPreprocessor'
        ],
        'cv': [
            'ImageClassificationMmcvPreprocessor',
            'ImageRestorationPreprocessor',
            'ControllableImageGenerationPreprocessor'
        ],
        'kws': ['WavToLists'],
        'tts': ['KanttsDataPreprocessor'],
        'multi_modal': [
            'DiffusionImageGenerationPreprocessor', 'OfaPreprocessor',
            'MPlugPreprocessor', 'HiTeAPreprocessor', 'MplugOwlPreprocessor',
            'ImageCaptioningClipInterrogatorPreprocessor'
        ],
        'nlp': [
            'DocumentSegmentationTransformersPreprocessor',
            'FaqQuestionAnsweringTransformersPreprocessor',
            'FillMaskPoNetPreprocessor',
            'FillMaskTransformersPreprocessor',
            'NLPTokenizerPreprocessorBase',
            'TextRankingTransformersPreprocessor',
            'RelationExtractionTransformersPreprocessor',
            'SentenceEmbeddingTransformersPreprocessor',
            'TextGenerationSentencePiecePreprocessor',
            'TextClassificationTransformersPreprocessor',
            'TokenClassificationTransformersPreprocessor',
            'TextErrorCorrectionPreprocessor',
            'WordAlignmentPreprocessor',
            'TextGenerationTransformersPreprocessor',
            'Tokenize',
            'TextGenerationT5Preprocessor',
            'WordSegmentationBlankSetToLabelPreprocessor',
            'MGLMSummarizationPreprocessor',
            'CodeGeeXPreprocessor',
            'ZeroShotClassificationTransformersPreprocessor',
            'TextGenerationJiebaPreprocessor',
            'SentencePiecePreprocessor',
            'NERPreprocessorViet',
            'NERPreprocessorThai',
            'WordSegmentationPreprocessorThai',
            'DialogIntentPredictionPreprocessor',
            'DialogModelingPreprocessor',
            'DialogStateTrackingPreprocessor',
            'ConversationalTextToSqlPreprocessor',
            'TableQuestionAnsweringPreprocessor',
            'TranslationEvaluationTransformersPreprocessor',
            'CanmtTranslationPreprocessor',
            'DialogueClassificationUsePreprocessor',
            'SiameseUiePreprocessor',
            'DialogueClassificationUsePreprocessor',
            'DocumentGroundedDialogGeneratePreprocessor',
            'DocumentGroundedDialogRetrievalPreprocessor',
            'DocumentGroundedDialogRerankPreprocessor',
            'MachineReadingComprehensionForNERPreprocessor',
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
