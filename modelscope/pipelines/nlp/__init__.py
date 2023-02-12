# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .automatic_post_editing_pipeline import AutomaticPostEditingPipeline
    from .conversational_text_to_sql_pipeline import ConversationalTextToSqlPipeline
    from .table_question_answering_pipeline import TableQuestionAnsweringPipeline
    from .dialog_intent_prediction_pipeline import DialogIntentPredictionPipeline
    from .dialog_modeling_pipeline import DialogModelingPipeline
    from .dialog_state_tracking_pipeline import DialogStateTrackingPipeline
    from .document_segmentation_pipeline import DocumentSegmentationPipeline
    from .extractive_summarization_pipeline import ExtractiveSummarizationPipeline
    from .fasttext_text_classification_pipeline import FasttextSequenceClassificationPipeline
    from .faq_question_answering_pipeline import FaqQuestionAnsweringPipeline
    from .feature_extraction_pipeline import FeatureExtractionPipeline
    from .fill_mask_pipeline import FillMaskPipeline
    from .information_extraction_pipeline import InformationExtractionPipeline
    from .interactive_translation_pipeline import InteractiveTranslationPipeline
    from .named_entity_recognition_pipeline import NamedEntityRecognitionPipeline
    from .text_ranking_pipeline import TextRankingPipeline
    from .sentence_embedding_pipeline import SentenceEmbeddingPipeline
    from .text_classification_pipeline import TextClassificationPipeline
    from .summarization_pipeline import SummarizationPipeline
    from .translation_quality_estimation_pipeline import TranslationQualityEstimationPipeline
    from .text_error_correction_pipeline import TextErrorCorrectionPipeline
    from .text_generation_pipeline import TextGenerationPipeline, TextGenerationT5Pipeline
    from .fid_dialogue_pipeline import FidDialoguePipeline
    from .token_classification_pipeline import TokenClassificationPipeline
    from .translation_pipeline import TranslationPipeline
    from .word_segmentation_pipeline import WordSegmentationPipeline, WordSegmentationThaiPipeline
    from .zero_shot_classification_pipeline import ZeroShotClassificationPipeline
    from .mglm_text_summarization_pipeline import MGLMTextSummarizationPipeline
    from .codegeex_code_translation_pipeline import CodeGeeXCodeTranslationPipeline
    from .codegeex_code_generation_pipeline import CodeGeeXCodeGenerationPipeline
    from .translation_evaluation_pipeline import TranslationEvaluationPipeline
    from .user_satisfaction_estimation_pipeline import UserSatisfactionEstimationPipeline
    from .document_grounded_dialog_generate_pipeline import DocumentGroundedDialogGeneratePipeline
    from .document_grounded_dialog_retrieval_pipeline import DocumentGroundedDialogRetrievalPipeline
    from .document_grounded_dialog_rerank_pipeline import DocumentGroundedDialogRerankPipeline

else:
    _import_structure = {
        'automatic_post_editing_pipeline': ['AutomaticPostEditingPipeline'],
        'conversational_text_to_sql_pipeline':
        ['ConversationalTextToSqlPipeline'],
        'dialog_intent_prediction_pipeline':
        ['DialogIntentPredictionPipeline'],
        'dialog_modeling_pipeline': ['DialogModelingPipeline'],
        'dialog_state_tracking_pipeline': ['DialogStateTrackingPipeline'],
        'fasttext_text_classification_pipeline':
        ['FasttextSequenceClassificationPipeline'],
        'document_segmentation_pipeline': ['DocumentSegmentationPipeline'],
        'extractive_summarization_pipeline':
        ['ExtractiveSummarizationPipeline'],
        'faq_question_answering_pipeline': ['FaqQuestionAnsweringPipeline'],
        'feature_extraction_pipeline': ['FeatureExtractionPipeline'],
        'fill_mask_pipeline': ['FillMaskPipeline'],
        'information_extraction_pipeline': ['InformationExtractionPipeline'],
        'interactive_translation_pipeline': ['InteractiveTranslationPipeline'],
        'named_entity_recognition_pipeline': [
            'NamedEntityRecognitionPipeline',
        ],
        'text_ranking_pipeline': ['TextRankingPipeline'],
        'sentence_embedding_pipeline': ['SentenceEmbeddingPipeline'],
        'summarization_pipeline': ['SummarizationPipeline'],
        'table_question_answering_pipeline':
        ['TableQuestionAnsweringPipeline'],
        'text_classification_pipeline': ['TextClassificationPipeline'],
        'text_error_correction_pipeline': ['TextErrorCorrectionPipeline'],
        'text_generation_pipeline':
        ['TextGenerationPipeline', 'TextGenerationT5Pipeline'],
        'fid_dialogue_pipeline': ['FidDialoguePipeline'],
        'token_classification_pipeline': ['TokenClassificationPipeline'],
        'translation_pipeline': ['TranslationPipeline'],
        'translation_quality_estimation_pipeline':
        ['TranslationQualityEstimationPipeline'],
        'word_segmentation_pipeline':
        ['WordSegmentationPipeline', 'WordSegmentationThaiPipeline'],
        'zero_shot_classification_pipeline':
        ['ZeroShotClassificationPipeline'],
        'mglm_text_summarization_pipeline': ['MGLMTextSummarizationPipeline'],
        'codegeex_code_translation_pipeline':
        ['CodeGeeXCodeTranslationPipeline'],
        'codegeex_code_generation_pipeline':
        ['CodeGeeXCodeGenerationPipeline'],
        'translation_evaluation_pipeline': ['TranslationEvaluationPipeline'],
        'user_satisfaction_estimation_pipeline':
        ['UserSatisfactionEstimationPipeline'],
        'document_grounded_dialog_generate_pipeline': [
            'DocumentGroundedDialogGeneratePipeline'
        ],
        'document_grounded_dialog_rerank_pipeline': [
            'DocumentGroundedDialogRerankPipeline'
        ],
        'document_grounded_dialog_retrieval_pipeline': [
            'DocumentGroundedDialogRetrievalPipeline'
        ]
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
