# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .dialog_intent_prediction_pipeline import DialogIntentPredictionPipeline
    from .dialog_modeling_pipeline import DialogModelingPipeline
    from .dialog_state_tracking_pipeline import DialogStateTrackingPipeline
    from .fill_mask_pipeline import FillMaskPipeline
    from .named_entity_recognition_pipeline import NamedEntityRecognitionPipeline
    from .nli_pipeline import NLIPipeline
    from .sentence_similarity_pipeline import SentenceSimilarityPipeline
    from .sentiment_classification_pipeline import SentimentClassificationPipeline
    from .sequence_classification_pipeline import SequenceClassificationPipeline
    from .text_generation_pipeline import TextGenerationPipeline
    from .translation_pipeline import TranslationPipeline
    from .word_segmentation_pipeline import WordSegmentationPipeline
    from .zero_shot_classification_pipeline import ZeroShotClassificationPipeline
    from .text_error_correction_pipeline import TextErrorCorrectionPipeline

else:
    _import_structure = {
        'dialog_intent_prediction_pipeline':
        ['DialogIntentPredictionPipeline'],
        'dialog_modeling_pipeline': ['DialogModelingPipeline'],
        'dialog_state_tracking_pipeline': ['DialogStateTrackingPipeline'],
        'fill_mask_pipeline': ['FillMaskPipeline'],
        'nli_pipeline': ['NLIPipeline'],
        'sentence_similarity_pipeline': ['SentenceSimilarityPipeline'],
        'sentiment_classification_pipeline':
        ['SentimentClassificationPipeline'],
        'sequence_classification_pipeline': ['SequenceClassificationPipeline'],
        'text_generation_pipeline': ['TextGenerationPipeline'],
        'word_segmentation_pipeline': ['WordSegmentationPipeline'],
        'zero_shot_classification_pipeline':
        ['ZeroShotClassificationPipeline'],
        'named_entity_recognition_pipeline':
        ['NamedEntityRecognitionPipeline'],
        'translation_pipeline': ['TranslationPipeline'],
        'text_error_correction_pipeline': ['TextErrorCorrectionPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
