try:
    from .action_recognition_pipeline import ActionRecognitionPipeline
    from .animal_recog_pipeline import AnimalRecogPipeline
    from .cmdssl_video_embedding_pipleline import CMDSSLVideoEmbeddingPipeline
except ModuleNotFoundError as e:
    if str(e) == "No module named 'torch'":
        pass
    else:
        raise ModuleNotFoundError(e)

try:
    from .image_cartoon_pipeline import ImageCartoonPipeline
    from .image_matting_pipeline import ImageMattingPipeline
    from .ocr_detection_pipeline import OCRDetectionPipeline
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        pass
    else:
        raise ModuleNotFoundError(e)
