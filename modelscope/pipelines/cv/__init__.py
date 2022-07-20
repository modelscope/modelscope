# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.error import TENSORFLOW_IMPORT_ERROR

try:
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recog_pipeline import AnimalRecogPipeline
    from .cmdssl_video_embedding_pipleline import CMDSSLVideoEmbeddingPipeline
    from .face_image_generation_pipeline import FaceImageGenerationPipeline
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
