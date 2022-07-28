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
        ['GEMMMultiModalEmbeddingPipeline']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
