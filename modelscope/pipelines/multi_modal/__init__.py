# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .asr_pipeline import AutomaticSpeechRecognitionPipeline
    from .diffusers_wrapped import (ChineseStableDiffusionPipeline,
                                    StableDiffusionPipeline)
    from .document_vl_embedding_pipeline import DocumentVLEmbeddingPipeline
    from .generative_multi_modal_embedding_pipeline import \
        GEMMMultiModalEmbeddingPipeline
    from .image_captioning_pipeline import ImageCaptioningPipeline
    from .mgeo_ranking_pipeline import MGeoRankingPipeline
    from .multi_modal_embedding_pipeline import MultiModalEmbeddingPipeline
    from .multimodal_dialogue_pipeline import MultimodalDialoguePipeline
    from .prost_text_video_retrieval_pipeline import \
        ProSTTextVideoRetrievalPipeline
    from .soonet_video_temporal_grounding_pipeline import \
        SOONetVideoTemporalGroundingPipeline
    from .text_to_image_synthesis_pipeline import TextToImageSynthesisPipeline
    from .text_to_video_synthesis_pipeline import TextToVideoSynthesisPipeline
    from .video_captioning_pipeline import VideoCaptioningPipeline
    from .video_multi_modal_embedding_pipeline import \
        VideoMultiModalEmbeddingPipeline
    from .visual_question_answering_pipeline import VisualQuestionAnsweringPipeline
    from .video_question_answering_pipeline import VideoQuestionAnsweringPipeline
    from .videocomposer_pipeline import VideoComposerPipeline
    from .text_to_image_freeu_pipeline import FreeUTextToImagePipeline
else:
    _import_structure = {
        'image_captioning_pipeline': ['ImageCaptioningPipeline'],
        'visual_entailment_pipeline': ['VisualEntailmentPipeline'],
        'visual_grounding_pipeline': ['VisualGroundingPipeline'],
        'multi_modal_embedding_pipeline': ['MultiModalEmbeddingPipeline'],
        'prost_text_video_retrieval_pipeline':
        ['ProSTTextVideoRetrievalPipeline'],
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
        ['StableDiffusionPipeline', 'ChineseStableDiffusionPipeline'],
        'soonet_video_temporal_grounding_pipeline':
        ['SOONetVideoTemporalGroundingPipeline'],
        'text_to_video_synthesis_pipeline': ['TextToVideoSynthesisPipeline'],
        'multimodal_dialogue_pipeline': ['MultimodalDialoguePipeline'],
        'videocomposer_pipeline': ['VideoComposerPipeline'],
        'text_to_image_freeu_pipeline': ['FreeUTextToImagePipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
