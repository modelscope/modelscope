# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .audio_noise_metric import AudioNoiseMetric
    from .base import Metric
    from .builder import METRICS, build_metric, task_default_metrics
    from .image_color_enhance_metric import ImageColorEnhanceMetric
    from .image_denoise_metric import ImageDenoiseMetric
    from .image_instance_segmentation_metric import \
        ImageInstanceSegmentationCOCOMetric
    from .image_portrait_enhancement_metric import ImagePortraitEnhancementMetric
    from .sequence_classification_metric import SequenceClassificationMetric
    from .text_generation_metric import TextGenerationMetric
    from .token_classification_metric import TokenClassificationMetric
    from .video_summarization_metric import VideoSummarizationMetric
    from .movie_scene_segmentation_metric import MovieSceneSegmentationMetric
    from .accuracy_metric import AccuracyMetric
    from .bleu_metric import BleuMetric

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
        'accuracy_metric': ['AccuracyMetric'],
        'bleu_metric': ['BleuMetric'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
