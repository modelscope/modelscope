# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .easycv_base import EasyCVBaseDataset
    from .builder import CUSTOM_DATASETS, build_custom_dataset
    from .torch_custom_dataset import TorchCustomDataset
    from .movie_scene_segmentation.movie_scene_segmentation_dataset import MovieSceneSegmentationDataset
    from .image_instance_segmentation_coco_dataset import ImageInstanceSegmentationCocoDataset
    from .gopro_image_deblurring_dataset import GoproImageDeblurringDataset
    from .language_guided_video_summarization_dataset import LanguageGuidedVideoSummarizationDataset
    from .mgeo_ranking_dataset import MGeoRankingDataset
    from .reds_image_deblurring_dataset import RedsImageDeblurringDataset
    from .text_ranking_dataset import TextRankingDataset
    from .veco_dataset import VecoDataset
    from .video_summarization_dataset import VideoSummarizationDataset
    from .audio import KWSDataset, KWSDataLoader, kws_nearfield_dataset, ASRDataset
    from .bad_image_detecting import BadImageDetectingDataset
    from .image_inpainting import ImageInpaintingDataset
    from .image_portrait_enhancement import ImagePortraitEnhancementDataset
    from .image_quality_assessment_degradation import ImageQualityAssessmentDegradationDataset
    from .image_quality_assmessment_mos import ImageQualityAssessmentMosDataset
    from .referring_video_object_segmentation import ReferringVideoObjectSegmentationDataset
    from .sidd_image_denoising import SiddImageDenoisingDataset
    from .video_frame_interpolation import VideoFrameInterpolationDataset
    from .video_stabilization import VideoStabilizationDataset
    from .video_super_resolution import VideoSuperResolutionDataset
    from .ocr_detection import DataLoader, ImageDataset, QuadMeasurer
    from .ocr_recognition_dataset import OCRRecognitionDataset
    from .image_colorization import ImageColorizationDataset
else:
    _import_structure = {
        'easycv_base': ['EasyCVBaseDataset'],
        'builder': ['CUSTOM_DATASETS', 'build_custom_dataset'],
        'torch_custom_dataset': ['TorchCustomDataset'],
        'movie_scene_segmentation_dataset': ['MovieSceneSegmentationDataset'],
        'image_instance_segmentation_coco_dataset':
        ['ImageInstanceSegmentationCocoDataset'],
        'gopro_image_deblurring_dataset': ['GoproImageDeblurringDataset'],
        'language_guided_video_summarization_dataset':
        ['LanguageGuidedVideoSummarizationDataset'],
        'mgeo_ranking_dataset': ['MGeoRankingDataset'],
        'reds_image_deblurring_dataset': ['RedsImageDeblurringDataset'],
        'text_ranking_dataset': ['TextRankingDataset'],
        'veco_dataset': ['VecoDataset'],
        'video_summarization_dataset': ['VideoSummarizationDataset'],
        'audio':
        ['KWSDataset', 'KWSDataLoader', 'kws_nearfield_dataset', 'ASRDataset'],
        'bad_image_detecting': ['BadImageDetectingDataset'],
        'image_inpainting': ['ImageInpaintingDataset'],
        'image_portrait_enhancement': ['ImagePortraitEnhancementDataset'],
        'image_quality_assessment_degradation':
        ['ImageQualityAssessmentDegradationDataset'],
        'image_quality_assmessment_mos': ['ImageQualityAssessmentMosDataset'],
        'referring_video_object_segmentation':
        ['ReferringVideoObjectSegmentationDataset'],
        'sidd_image_denoising': ['SiddImageDenoisingDataset'],
        'video_frame_interpolation': ['VideoFrameInterpolationDataset'],
        'video_stabilization': ['VideoStabilizationDataset'],
        'video_super_resolution': ['VideoSuperResolutionDataset'],
        'ocr_detection': ['DataLoader', 'ImageDataset', 'QuadMeasurer'],
        'ocr_recognition_dataset': ['OCRRecognitionDataset'],
        'image_colorization': ['ImageColorizationDataset'],
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
