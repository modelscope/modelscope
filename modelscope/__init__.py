# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .version import __release_datetime__, __version__
    from .trainers import EpochBasedTrainer, TrainingArgs, build_dataset_from_file
    from .trainers import Hook, Priority
    from .exporters import Exporter
    from .exporters import TfModelExporter
    from .exporters import TorchModelExporter
    from .hub.api import HubApi
    from .hub.snapshot_download import snapshot_download
    from .hub.push_to_hub import push_to_hub, push_to_hub_async
    from .hub.check_model import check_model_is_id, check_local_model_is_latest
    from .metrics import AudioNoiseMetric, Metric, task_default_metrics, ImageColorEnhanceMetric, ImageDenoiseMetric, \
        ImageInstanceSegmentationCOCOMetric, ImagePortraitEnhancementMetric, SequenceClassificationMetric, \
        TextGenerationMetric, TokenClassificationMetric, VideoSummarizationMetric, MovieSceneSegmentationMetric, \
        AccuracyMetric, BleuMetric, ImageInpaintingMetric, ReferringVideoObjectSegmentationMetric, \
        VideoFrameInterpolationMetric, VideoStabilizationMetric, VideoSuperResolutionMetric, PplMetric, \
        ImageQualityAssessmentDegradationMetric, ImageQualityAssessmentMosMetric, TextRankingMetric, \
        LossMetric, ImageColorizationMetric, OCRRecognitionMetric
    from .models import Model, TorchModel
    from .preprocessors import Preprocessor
    from .pipelines import Pipeline, pipeline
    from .utils.hub import read_config, create_model_if_not_exist
    from .utils.logger import get_logger
    from .utils.hf_util import AutoConfig, GenerationConfig
    from .utils.hf_util import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from .utils.hf_util import AutoTokenizer
    from .msdatasets import MsDataset

else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'trainers': [
            'EpochBasedTrainer', 'TrainingArgs', 'Hook', 'Priority',
            'build_dataset_from_file'
        ],
        'exporters': [
            'Exporter',
            'TfModelExporter',
            'TorchModelExporter',
        ],
        'hub.api': ['HubApi'],
        'hub.snapshot_download': ['snapshot_download'],
        'hub.push_to_hub': ['push_to_hub', 'push_to_hub_async'],
        'hub.check_model':
        ['check_model_is_id', 'check_local_model_is_latest'],
        'metrics': [
            'AudioNoiseMetric', 'Metric', 'task_default_metrics',
            'ImageColorEnhanceMetric', 'ImageDenoiseMetric',
            'ImageInstanceSegmentationCOCOMetric',
            'ImagePortraitEnhancementMetric', 'SequenceClassificationMetric',
            'TextGenerationMetric', 'TokenClassificationMetric',
            'VideoSummarizationMetric', 'MovieSceneSegmentationMetric',
            'AccuracyMetric', 'BleuMetric', 'ImageInpaintingMetric',
            'ReferringVideoObjectSegmentationMetric',
            'VideoFrameInterpolationMetric', 'VideoStabilizationMetric',
            'VideoSuperResolutionMetric', 'PplMetric',
            'ImageQualityAssessmentDegradationMetric',
            'ImageQualityAssessmentMosMetric', 'TextRankingMetric',
            'LossMetric', 'ImageColorizationMetric', 'OCRRecognitionMetric'
        ],
        'models': ['Model', 'TorchModel'],
        'preprocessors': ['Preprocessor'],
        'pipelines': ['Pipeline', 'pipeline'],
        'utils.hub': ['read_config', 'create_model_if_not_exist'],
        'utils.logger': ['get_logger'],
        'utils.hf_util': [
            'AutoConfig', 'GenerationConfig', 'AutoModel',
            'AutoModelForCausalLM', 'AutoModelForSeq2SeqLM', 'AutoTokenizer'
        ],
        'msdatasets': ['MsDataset']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
