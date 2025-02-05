# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import (LazyImportModule,
                                           is_transformers_available)

if TYPE_CHECKING:
    from .exporters import Exporter, TfModelExporter, TorchModelExporter
    from .hub.api import HubApi
    from .hub.check_model import check_local_model_is_latest, check_model_is_id
    from .hub.push_to_hub import push_to_hub, push_to_hub_async
    from .hub.snapshot_download import snapshot_download, dataset_snapshot_download
    from .hub.file_download import model_file_download, dataset_file_download

    from .metrics import (
        AccuracyMetric, AudioNoiseMetric, BleuMetric, ImageColorEnhanceMetric,
        ImageColorizationMetric, ImageDenoiseMetric, ImageInpaintingMetric,
        ImageInstanceSegmentationCOCOMetric, ImagePortraitEnhancementMetric,
        ImageQualityAssessmentDegradationMetric,
        ImageQualityAssessmentMosMetric, LossMetric, Metric,
        MovieSceneSegmentationMetric, OCRRecognitionMetric, PplMetric,
        ReferringVideoObjectSegmentationMetric, SequenceClassificationMetric,
        TextGenerationMetric, TextRankingMetric, TokenClassificationMetric,
        VideoFrameInterpolationMetric, VideoStabilizationMetric,
        VideoSummarizationMetric, VideoSuperResolutionMetric,
        task_default_metrics)
    from .models import Model, TorchModel
    from .msdatasets import MsDataset
    from .pipelines import Pipeline, pipeline
    from .preprocessors import Preprocessor
    from .trainers import (EpochBasedTrainer, Hook, Priority, TrainingArgs,
                           build_dataset_from_file)
    from .utils.constant import Tasks
    from .utils.hf_util import patch_hub, patch_context, unpatch_hub
    if is_transformers_available():
        from .utils.hf_util import (
            AutoModel, AutoProcessor, AutoFeatureExtractor, GenerationConfig,
            AutoConfig, GPTQConfig, AwqConfig, BitsAndBytesConfig,
            AutoModelForCausalLM, AutoModelForSeq2SeqLM,
            AutoModelForVision2Seq, AutoModelForSequenceClassification,
            AutoModelForTokenClassification, AutoModelForImageClassification,
            AutoModelForImageTextToText,
            AutoModelForZeroShotImageClassification,
            AutoModelForKeypointDetection,
            AutoModelForDocumentQuestionAnswering,
            AutoModelForSemanticSegmentation,
            AutoModelForUniversalSegmentation,
            AutoModelForInstanceSegmentation, AutoModelForObjectDetection,
            AutoModelForZeroShotObjectDetection,
            AutoModelForAudioClassification, AutoModelForSpeechSeq2Seq,
            AutoModelForMaskedImageModeling,
            AutoModelForVisualQuestionAnswering,
            AutoModelForTableQuestionAnswering, AutoModelForImageToImage,
            AutoModelForImageSegmentation, AutoModelForQuestionAnswering,
            AutoModelForMaskedLM, AutoTokenizer, AutoModelForMaskGeneration,
            AutoModelForPreTraining, AutoModelForTextEncoding,
            AutoImageProcessor, BatchFeature, Qwen2VLForConditionalGeneration,
            T5EncoderModel, Qwen2_5_VLForConditionalGeneration, LlamaModel,
            LlamaPreTrainedModel, LlamaForCausalLM)
    else:
        print(
            'transformer is not installed, please install it if you want to use related modules'
        )
    from .utils.hub import create_model_if_not_exist, read_config
    from .utils.logger import get_logger
    from .version import __release_datetime__, __version__

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
        'hub.snapshot_download':
        ['snapshot_download', 'dataset_snapshot_download'],
        'hub.file_download': ['model_file_download', 'dataset_file_download'],
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
        'utils.constant': ['Tasks'],
        'msdatasets': ['MsDataset']
    }

    from modelscope.utils import hf_util

    extra_objects = {}
    attributes = dir(hf_util)
    imports = [attr for attr in attributes if not attr.startswith('__')]
    for _import in imports:
        extra_objects[_import] = getattr(hf_util, _import)

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=extra_objects,
    )
