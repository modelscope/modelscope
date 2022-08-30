# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .conversational_text_to_sql_pipeline import ConversationalTextToSqlPipeline
    from .dialog_intent_prediction_pipeline import DialogIntentPredictionPipeline
    from .dialog_modeling_pipeline import DialogModelingPipeline
    from .dialog_state_tracking_pipeline import DialogStateTrackingPipeline
    from .document_segmentation_pipeline import DocumentSegmentationPipeline
    from .fill_mask_pipeline import FillMaskPipeline
    from .named_entity_recognition_pipeline import NamedEntityRecognitionPipeline
    from .pair_sentence_classification_pipeline import PairSentenceClassificationPipeline
    from .single_sentence_classification_pipeline import SingleSentenceClassificationPipeline
    from .sequence_classification_pipeline import SequenceClassificationPipeline
    from .text_generation_pipeline import TextGenerationPipeline
    from .translation_pipeline import TranslationPipeline
    from .word_segmentation_pipeline import WordSegmentationPipeline
    from .zero_shot_classification_pipeline import ZeroShotClassificationPipeline
    from .summarization_pipeline import SummarizationPipeline
    from .text_classification_pipeline import TextClassificationPipeline
    from .text_error_correction_pipeline import TextErrorCorrectionPipeline
    from .faq_question_answering_pipeline import FaqQuestionAnsweringPipeline

else:
    _import_structure = {
        'conversational_text_to_sql_pipeline':
        ['ConversationalTextToSqlPipeline'],
        'dialog_intent_prediction_pipeline':
        ['DialogIntentPredictionPipeline'],
        'dialog_modeling_pipeline': ['DialogModelingPipeline'],
        'dialog_state_tracking_pipeline': ['DialogStateTrackingPipeline'],
        'document_segmentation_pipeline': ['DocumentSegmentationPipeline'],
        'fill_mask_pipeline': ['FillMaskPipeline'],
        'single_sentence_classification_pipeline':
        ['SingleSentenceClassificationPipeline'],
        'pair_sentence_classification_pipeline':
        ['PairSentenceClassificationPipeline'],
        'sequence_classification_pipeline': ['SequenceClassificationPipeline'],
        'text_generation_pipeline': ['TextGenerationPipeline'],
        'word_segmentation_pipeline': ['WordSegmentationPipeline'],
        'zero_shot_classification_pipeline':
        ['ZeroShotClassificationPipeline'],
        'named_entity_recognition_pipeline':
        ['NamedEntityRecognitionPipeline'],
        'translation_pipeline': ['TranslationPipeline'],
        'summarization_pipeline': ['SummarizationPipeline'],
        'text_classification_pipeline': ['TextClassificationPipeline'],
        'text_error_correction_pipeline': ['TextErrorCorrectionPipeline'],
        'faq_question_answering_pipeline': ['FaqQuestionAnsweringPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
