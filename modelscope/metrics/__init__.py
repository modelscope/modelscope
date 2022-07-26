from .base import Metric
from .builder import METRICS, build_metric, task_default_metrics
from .image_color_enhance_metric import ImageColorEnhanceMetric
from .image_instance_segmentation_metric import \
    ImageInstanceSegmentationCOCOMetric
from .sequence_classification_metric import SequenceClassificationMetric
from .text_generation_metric import TextGenerationMetric
