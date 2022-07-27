# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recog_pipeline import AnimalRecogPipeline
    from .cmdssl_video_embedding_pipleline import CMDSSLVideoEmbeddingPipeline
    from .image_classification_pipeline import GeneralImageClassificationPipeline
    from .face_image_generation_pipeline import FaceImageGenerationPipeline
    from .image_cartoon_pipeline import ImageCartoonPipeline
    from .image_denoise_pipeline import ImageDenoisePipeline
    from .image_color_enhance_pipeline import ImageColorEnhancePipeline
    from .image_colorization_pipeline import ImageColorizationPipeline
    from .image_instance_segmentation_pipeline import ImageInstanceSegmentationPipeline
    from .image_matting_pipeline import ImageMattingPipeline
    from .image_super_resolution_pipeline import ImageSuperResolutionPipeline
    from .style_transfer_pipeline import StyleTransferPipeline
    from .ocr_detection_pipeline import OCRDetectionPipeline
    from .virtual_tryon_pipeline import VirtualTryonPipeline
else:
    _import_structure = {
        'action_recognition_pipeline': ['ActionRecognitionPipeline'],
        'animal_recog_pipeline': ['AnimalRecogPipeline'],
        'cmdssl_video_embedding_pipleline': ['CMDSSLVideoEmbeddingPipeline'],
        'image_classification_pipeline':
        ['GeneralImageClassificationPipeline'],
        'image_color_enhance_pipeline': ['ImageColorEnhancePipeline'],
        'virtual_tryon_pipeline': ['VirtualTryonPipeline'],
        'image_colorization_pipeline': ['ImageColorizationPipeline'],
        'image_super_resolution_pipeline': ['ImageSuperResolutionPipeline'],
        'image_denoise_pipeline': ['ImageDenoisePipeline'],
        'face_image_generation_pipeline': ['FaceImageGenerationPipeline'],
        'image_cartoon_pipeline': ['ImageCartoonPipeline'],
        'image_matting_pipeline': ['ImageMattingPipeline'],
        'style_transfer_pipeline': ['StyleTransferPipeline'],
        'ocr_detection_pipeline': ['OCRDetectionPipeline'],
        'image_instance_segmentation_pipeline':
        ['ImageInstanceSegmentationPipeline'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
