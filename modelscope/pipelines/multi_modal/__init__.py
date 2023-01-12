# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .generative_multi_modal_embedding_pipeline import GEMMMultiModalEmbeddingPipeline
    from .image_captioning_pipeline import ImageCaptioningPipeline
    from .visual_entailment_pipeline import VisualEntailmentPipeline
    from .visual_grounding_pipeline import VisualGroundingPipeline
    from .multi_modal_embedding_pipeline import MultiModalEmbeddingPipeline
    from .text_to_image_synthesis_pipeline import TextToImageSynthesisPipeline
    from .video_multi_modal_embedding_pipeline import \
        VideoMultiModalEmbeddingPipeline
    from .visual_question_answering_pipeline import VisualQuestionAnsweringPipeline
    from .asr_pipeline import AutomaticSpeechRecognitionPipeline
    from .mgeo_ranking_pipeline import MGeoRankingPipeline
    from .document_vl_embedding_pipeline import DocumentVLEmbeddingPipeline
    from .video_captioning_pipeline import VideoCaptioningPipeline
    from .video_question_answering_pipeline import VideoQuestionAnsweringPipeline
    from .diffusers_wrapped import StableDiffusionWrapperPipeline, ChineseStableDiffusionPipeline
else:
    _import_structure = {
        'image_captioning_pipeline': ['ImageCaptioningPipeline'],
        'visual_entailment_pipeline': ['VisualEntailmentPipeline'],
        'visual_grounding_pipeline': ['VisualGroundingPipeline'],
        'multi_modal_embedding_pipeline': ['MultiModalEmbeddingPipeline'],
        'text_to_image_synthesis_pipeline': ['TextToImageSynthesisPipeline'],
        'visual_question_answering_pipeline':
        ['VisualQuestionAnsweringPipeline'],
        'video_multi_modal_embedding_pipeline':
        ['VideoMultiModalEmbeddingPipeline'],
        'generative_multi_modal_embedding_pipeline':
        ['GEMMMultiModalEmbeddingPipeline'],
        'asr_pipeline': ['AutomaticSpeechRecognitionPipeline'],
        'mgeo_ranking_pipeline': ['MGeoRankingPipeline'],
        'document_vl_embedding_pipeline': ['DocumentVLEmbeddingPipeline'],
        'video_captioning_pipeline': ['VideoCaptioningPipeline'],
        'video_question_answering_pipeline':
        ['VideoQuestionAnsweringPipeline'],
        'diffusers_wrapped':
        ['StableDiffusionWrapperPipeline', 'ChineseStableDiffusionPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
