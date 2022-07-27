# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.error import TENSORFLOW_IMPORT_ERROR

try:
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recog_pipeline import AnimalRecogPipeline
    from .cmdssl_video_embedding_pipleline import CMDSSLVideoEmbeddingPipeline
    from .image_denoise_pipeline import ImageDenoisePipeline
    from .image_color_enhance_pipeline import ImageColorEnhancePipeline
    from .virtual_tryon_pipeline import VirtualTryonPipeline
    from .image_colorization_pipeline import ImageColorizationPipeline
    from .image_super_resolution_pipeline import ImageSuperResolutionPipeline
    from .face_image_generation_pipeline import FaceImageGenerationPipeline
    from .image_instance_segmentation_pipeline import ImageInstanceSegmentationPipeline
except ModuleNotFoundError as e:
    if str(e) == "No module named 'torch'":
        pass
    else:
        raise ModuleNotFoundError(e)

try:
    from .image_cartoon_pipeline import ImageCartoonPipeline
    from .image_matting_pipeline import ImageMattingPipeline
    from .style_transfer_pipeline import StyleTransferPipeline
    from .ocr_detection_pipeline import OCRDetectionPipeline
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        print(
            TENSORFLOW_IMPORT_ERROR.format(
                'image-cartoon image-matting ocr-detection style-transfer'))
    else:
        raise ModuleNotFoundError(e)
