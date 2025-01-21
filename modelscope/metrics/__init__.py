# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .accuracy_metric import AccuracyMetric
    from .audio_noise_metric import AudioNoiseMetric
    from .base import Metric
    from .bleu_metric import BleuMetric
    from .builder import METRICS, build_metric, task_default_metrics
    from .image_color_enhance_metric import ImageColorEnhanceMetric
    from .image_colorization_metric import ImageColorizationMetric
    from .image_denoise_metric import ImageDenoiseMetric
    from .image_inpainting_metric import ImageInpaintingMetric
    from .image_instance_segmentation_metric import \
        ImageInstanceSegmentationCOCOMetric
    from .image_portrait_enhancement_metric import \
        ImagePortraitEnhancementMetric
    from .image_quality_assessment_degradation_metric import \
        ImageQualityAssessmentDegradationMetric
    from .image_quality_assessment_mos_metric import \
        ImageQualityAssessmentMosMetric
    from .loss_metric import LossMetric
    from .movie_scene_segmentation_metric import MovieSceneSegmentationMetric
    from .ocr_recognition_metric import OCRRecognitionMetric
    from .ppl_metric import PplMetric
    from .referring_video_object_segmentation_metric import \
        ReferringVideoObjectSegmentationMetric
    from .sequence_classification_metric import SequenceClassificationMetric
    from .text_generation_metric import TextGenerationMetric
    from .text_ranking_metric import TextRankingMetric
    from .token_classification_metric import TokenClassificationMetric
    from .translation_evaluation_metric import TranslationEvaluationMetric
    from .video_frame_interpolation_metric import VideoFrameInterpolationMetric
    from .video_stabilization_metric import VideoStabilizationMetric
    from .video_summarization_metric import VideoSummarizationMetric
    from .video_super_resolution_metric.video_super_resolution_metric import \
        VideoSuperResolutionMetric
else:
    _import_structure = {
        'audio_noise_metric': ['AudioNoiseMetric'],
        'base': ['Metric'],
        'builder': ['METRICS', 'build_metric', 'task_default_metrics'],
        'image_color_enhance_metric': ['ImageColorEnhanceMetric'],
        'image_denoise_metric': ['ImageDenoiseMetric'],
        'image_instance_segmentation_metric':
        ['ImageInstanceSegmentationCOCOMetric'],
        'image_portrait_enhancement_metric':
        ['ImagePortraitEnhancementMetric'],
        'sequence_classification_metric': ['SequenceClassificationMetric'],
        'text_generation_metric': ['TextGenerationMetric'],
        'token_classification_metric': ['TokenClassificationMetric'],
        'video_summarization_metric': ['VideoSummarizationMetric'],
        'movie_scene_segmentation_metric': ['MovieSceneSegmentationMetric'],
        'image_inpainting_metric': ['ImageInpaintingMetric'],
        'accuracy_metric': ['AccuracyMetric'],
        'bleu_metric': ['BleuMetric'],
        'referring_video_object_segmentation_metric':
        ['ReferringVideoObjectSegmentationMetric'],
        'video_frame_interpolation_metric': ['VideoFrameInterpolationMetric'],
        'video_stabilization_metric': ['VideoStabilizationMetric'],
        'ppl_metric': ['PplMetric'],
        'image_quality_assessment_degradation_metric':
        ['ImageQualityAssessmentDegradationMetric'],
        'image_quality_assessment_mos_metric':
        ['ImageQualityAssessmentMosMetric'],
        'text_ranking_metric': ['TextRankingMetric'],
        'loss_metric': ['LossMetric'],
        'image_colorization_metric': ['ImageColorizationMetric'],
        'ocr_recognition_metric': ['OCRRecognitionMetric'],
        'translation_evaluation_metric': ['TranslationEvaluationMetric'],
        'video_super_resolution_metric.video_super_resolution_metric':
        ['VideoSuperResolutionMetric'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
