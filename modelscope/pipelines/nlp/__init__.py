# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .conversational_text_to_sql_pipeline import ConversationalTextToSqlPipeline
    from .dialog_intent_prediction_pipeline import DialogIntentPredictionPipeline
    from .dialog_modeling_pipeline import DialogModelingPipeline
    from .dialog_state_tracking_pipeline import DialogStateTrackingPipeline
    from .document_segmentation_pipeline import DocumentSegmentationPipeline
    from .faq_question_answering_pipeline import FaqQuestionAnsweringPipeline
    from .fill_mask_pipeline import FillMaskPipeline
    from .fill_mask_ponet_pipeline import FillMaskPoNetPreprocessor
    from .information_extraction_pipeline import InformationExtractionPipeline
    from .named_entity_recognition_pipeline import NamedEntityRecognitionPipeline
    from .pair_sentence_classification_pipeline import PairSentenceClassificationPipeline
    from .single_sentence_classification_pipeline import SingleSentenceClassificationPipeline
    from .sequence_classification_pipeline import SequenceClassificationPipeline
    from .summarization_pipeline import SummarizationPipeline
    from .text_classification_pipeline import TextClassificationPipeline
    from .text_error_correction_pipeline import TextErrorCorrectionPipeline
    from .text_generation_pipeline import TextGenerationPipeline
    from .token_classification_pipeline import TokenClassificationPipeline
    from .translation_pipeline import TranslationPipeline
    from .word_segmentation_pipeline import WordSegmentationPipeline
    from .zero_shot_classification_pipeline import ZeroShotClassificationPipeline
    from .passage_ranking_pipeline import PassageRankingPipeline
    from .sentence_embedding_pipeline import SentenceEmbeddingPipeline
else:
    _import_structure = {
        'conversational_text_to_sql_pipeline':
        ['ConversationalTextToSqlPipeline'],
        'dialog_intent_prediction_pipeline':
        ['DialogIntentPredictionPipeline'],
        'dialog_modeling_pipeline': ['DialogModelingPipeline'],
        'dialog_state_tracking_pipeline': ['DialogStateTrackingPipeline'],
        'document_segmentation_pipeline': ['DocumentSegmentationPipeline'],
        'faq_question_answering_pipeline': ['FaqQuestionAnsweringPipeline'],
        'fill_mask_pipeline': ['FillMaskPipeline'],
        'fill_mask_ponet_pipeline': ['FillMaskPoNetPipeline'],
        'named_entity_recognition_pipeline':
        ['NamedEntityRecognitionPipeline'],
        'information_extraction_pipeline': ['InformationExtractionPipeline'],
        'pair_sentence_classification_pipeline':
        ['PairSentenceClassificationPipeline'],
        'sequence_classification_pipeline': ['SequenceClassificationPipeline'],
        'single_sentence_classification_pipeline':
        ['SingleSentenceClassificationPipeline'],
        'summarization_pipeline': ['SummarizationPipeline'],
        'text_classification_pipeline': ['TextClassificationPipeline'],
        'text_error_correction_pipeline': ['TextErrorCorrectionPipeline'],
        'text_generation_pipeline': ['TextGenerationPipeline'],
        'token_classification_pipeline': ['TokenClassificationPipeline'],
        'translation_pipeline': ['TranslationPipeline'],
        'word_segmentation_pipeline': ['WordSegmentationPipeline'],
        'zero_shot_classification_pipeline':
        ['ZeroShotClassificationPipeline'],
        'passage_ranking_pipeline': ['PassageRankingPipeline'],
        'sentence_embedding_pipeline': ['SentenceEmbeddingPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
