# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recognition_pipeline import AnimalRecognitionPipeline
    from .cmdssl_video_embedding_pipeline import CMDSSLVideoEmbeddingPipeline
    from .image_detection_pipeline import ImageDetectionPipeline
    from .face_detection_pipeline import FaceDetectionPipeline
    from .face_recognition_pipeline import FaceRecognitionPipeline
    from .face_image_generation_pipeline import FaceImageGenerationPipeline
    from .image_classification_pipeline import ImageClassificationPipeline
    from .image_cartoon_pipeline import ImageCartoonPipeline
    from .image_classification_pipeline import GeneralImageClassificationPipeline
    from .image_denoise_pipeline import ImageDenoisePipeline
    from .image_color_enhance_pipeline import ImageColorEnhancePipeline
    from .image_colorization_pipeline import ImageColorizationPipeline
    from .image_instance_segmentation_pipeline import ImageInstanceSegmentationPipeline
    from .image_matting_pipeline import ImageMattingPipeline
    from .image_style_transfer_pipeline import ImageStyleTransferPipeline
    from .image_super_resolution_pipeline import ImageSuperResolutionPipeline
    from .image_to_image_generation_pipeline import Image2ImageGenerationePipeline
    from .image_to_image_translation_pipeline import Image2ImageTranslationPipeline
    from .product_retrieval_embedding_pipeline import ProductRetrievalEmbeddingPipeline
    from .live_category_pipeline import LiveCategoryPipeline
    from .ocr_detection_pipeline import OCRDetectionPipeline
    from .video_category_pipeline import VideoCategoryPipeline
    from .virtual_tryon_pipeline import VirtualTryonPipeline
else:
    _import_structure = {
        'action_recognition_pipeline': ['ActionRecognitionPipeline'],
        'animal_recognition_pipeline': ['AnimalRecognitionPipeline'],
        'cmdssl_video_embedding_pipeline': ['CMDSSLVideoEmbeddingPipeline'],
        'image_detection_pipeline': ['ImageDetectionPipeline'],
        'face_detection_pipeline': ['FaceDetectionPipeline'],
        'face_image_generation_pipeline': ['FaceImageGenerationPipeline'],
        'face_recognition_pipeline': ['FaceRecognitionPipeline'],
        'image_classification_pipeline':
        ['GeneralImageClassificationPipeline', 'ImageClassificationPipeline'],
        'image_cartoon_pipeline': ['ImageCartoonPipeline'],
        'image_denoise_pipeline': ['ImageDenoisePipeline'],
        'image_color_enhance_pipeline': ['ImageColorEnhancePipeline'],
        'image_colorization_pipeline': ['ImageColorizationPipeline'],
        'image_instance_segmentation_pipeline':
        ['ImageInstanceSegmentationPipeline'],
        'image_matting_pipeline': ['ImageMattingPipeline'],
        'image_style_transfer_pipeline': ['ImageStyleTransferPipeline'],
        'image_super_resolution_pipeline': ['ImageSuperResolutionPipeline'],
        'image_to_image_translation_pipeline':
        ['Image2ImageTranslationPipeline'],
        'product_retrieval_embedding_pipeline':
        ['ProductRetrievalEmbeddingPipeline'],
        'live_category_pipeline': ['LiveCategoryPipeline'],
        'image_to_image_generation_pipeline':
        ['Image2ImageGenerationePipeline'],
        'ocr_detection_pipeline': ['OCRDetectionPipeline'],
        'video_category_pipeline': ['VideoCategoryPipeline'],
        'virtual_tryon_pipeline': ['VirtualTryonPipeline'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
