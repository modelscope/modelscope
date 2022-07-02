try:
    from .image_captioning_pipeline import ImageCaptionPipeline
    from .multi_modal_embedding_pipeline import MultiModalEmbeddingPipeline
    from .text_to_image_synthesis_pipeline import TextToImageSynthesisPipeline
    from .visual_question_answering_pipeline import VisualQuestionAnsweringPipeline
except ModuleNotFoundError as e:
    if str(e) == "No module named 'torch'":
        pass
    else:
        raise ModuleNotFoundError(e)
