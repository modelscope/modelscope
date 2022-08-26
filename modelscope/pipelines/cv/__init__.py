# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recognition_pipeline import AnimalRecognitionPipeline
    from .body_2d_keypoints_pipeline import Body2DKeypointsPipeline
    from .cmdssl_video_embedding_pipeline import CMDSSLVideoEmbeddingPipeline
    from .crowd_counting_pipeline import CrowdCountingPipeline
    from .image_detection_pipeline import ImageDetectionPipeline
    from .image_salient_detection_pipeline import ImageSalientDetectionPipeline
    from .face_detection_pipeline import FaceDetectionPipeline
    from .face_image_generation_pipeline import FaceImageGenerationPipeline
    from .face_recognition_pipeline import FaceRecognitionPipeline
    from .general_recognition_pipeline import GeneralRecognitionPipeline
    from .image_cartoon_pipeline import ImageCartoonPipeline
    from .image_classification_pipeline import GeneralImageClassificationPipeline
    from .image_color_enhance_pipeline import ImageColorEnhancePipeline
    from .image_colorization_pipeline import ImageColorizationPipeline
    from .image_classification_pipeline import ImageClassificationPipeline
    from .image_denoise_pipeline import ImageDenoisePipeline
    from .image_instance_segmentation_pipeline import ImageInstanceSegmentationPipeline
    from .image_matting_pipeline import ImageMattingPipeline
    from .image_panoptic_segmentation_pipeline import ImagePanopticSegmentationPipeline
    from .image_portrait_enhancement_pipeline import ImagePortraitEnhancementPipeline
    from .image_reid_person_pipeline import ImageReidPersonPipeline
    from .image_semantic_segmentation_pipeline import ImageSemanticSegmentationPipeline
    from .image_style_transfer_pipeline import ImageStyleTransferPipeline
    from .image_super_resolution_pipeline import ImageSuperResolutionPipeline
    from .image_to_image_generate_pipeline import Image2ImageGenerationPipeline
    from .image_to_image_translation_pipeline import Image2ImageTranslationPipeline
    from .product_retrieval_embedding_pipeline import ProductRetrievalEmbeddingPipeline
    from .live_category_pipeline import LiveCategoryPipeline
    from .ocr_detection_pipeline import OCRDetectionPipeline
    from .ocr_recognition_pipeline import OCRRecognitionPipeline
    from .skin_retouching_pipeline import SkinRetouchingPipeline
    from .tinynas_classification_pipeline import TinynasClassificationPipeline
    from .video_category_pipeline import VideoCategoryPipeline
    from .virtual_try_on_pipeline import VirtualTryonPipeline
    from .easycv_pipelines import EasyCVDetectionPipeline, EasyCVSegmentationPipeline
else:
    _import_structure = {
        'action_recognition_pipeline': ['ActionRecognitionPipeline'],
        'animal_recognition_pipeline': ['AnimalRecognitionPipeline'],
        'body_2d_keypoints_pipeline': ['Body2DKeypointsPipeline'],
        'cmdssl_video_embedding_pipeline': ['CMDSSLVideoEmbeddingPipeline'],
        'crowd_counting_pipeline': ['CrowdCountingPipeline'],
        'image_detection_pipeline': ['ImageDetectionPipeline'],
        'image_salient_detection_pipeline': ['ImageSalientDetectionPipeline'],
        'face_detection_pipeline': ['FaceDetectionPipeline'],
        'face_image_generation_pipeline': ['FaceImageGenerationPipeline'],
        'face_recognition_pipeline': ['FaceRecognitionPipeline'],
        'general_recognition_pipeline': ['GeneralRecognitionPipeline'],
        'image_classification_pipeline':
        ['GeneralImageClassificationPipeline', 'ImageClassificationPipeline'],
        'image_cartoon_pipeline': ['ImageCartoonPipeline'],
        'image_denoise_pipeline': ['ImageDenoisePipeline'],
        'image_color_enhance_pipeline': ['ImageColorEnhancePipeline'],
        'image_colorization_pipeline': ['ImageColorizationPipeline'],
        'image_instance_segmentation_pipeline':
        ['ImageInstanceSegmentationPipeline'],
        'image_matting_pipeline': ['ImageMattingPipeline'],
        'image_panoptic_segmentation_pipeline':
        ['ImagePanopticSegmentationPipeline'],
        'image_portrait_enhancement_pipeline':
        ['ImagePortraitEnhancementPipeline'],
        'image_reid_person_pipeline': ['ImageReidPersonPipeline'],
        'image_semantic_segmentation_pipeline':
        ['ImageSemanticSegmentationPipeline'],
        'image_style_transfer_pipeline': ['ImageStyleTransferPipeline'],
        'image_super_resolution_pipeline': ['ImageSuperResolutionPipeline'],
        'image_to_image_translation_pipeline':
        ['Image2ImageTranslationPipeline'],
        'product_retrieval_embedding_pipeline':
        ['ProductRetrievalEmbeddingPipeline'],
        'live_category_pipeline': ['LiveCategoryPipeline'],
        'image_to_image_generation_pipeline':
        ['Image2ImageGenerationPipeline'],
        'ocr_detection_pipeline': ['OCRDetectionPipeline'],
        'ocr_recognition_pipeline': ['OCRRecognitionPipeline'],
        'skin_retouching_pipeline': ['SkinRetouchingPipeline'],
        'tinynas_classification_pipeline': ['TinynasClassificationPipeline'],
        'video_category_pipeline': ['VideoCategoryPipeline'],
        'virtual_try_on_pipeline': ['VirtualTryonPipeline'],
        'easycv_pipeline':
        ['EasyCVDetectionPipeline', 'EasyCVSegmentationPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
