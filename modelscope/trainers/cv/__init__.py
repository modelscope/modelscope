# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .image_instance_segmentation_trainer import \
        ImageInstanceSegmentationTrainer
    from .image_portrait_enhancement_trainer import ImagePortraitEnhancementTrainer
    from .movie_scene_segmentation_trainer import MovieSceneSegmentationTrainer
    from .image_inpainting_trainer import ImageInpaintingTrainer
    from .referring_video_object_segmentation_trainer import ReferringVideoObjectSegmentationTrainer
    from .image_defrcn_fewshot_detection_trainer import ImageDefrcnFewshotTrainer
    from .cartoon_translation_trainer import CartoonTranslationTrainer
    from .nerf_recon_acc_trainer import NeRFReconAccTrainer

else:
    _import_structure = {
        'image_instance_segmentation_trainer':
        ['ImageInstanceSegmentationTrainer'],
        'image_portrait_enhancement_trainer':
        ['ImagePortraitEnhancementTrainer'],
        'movie_scene_segmentation_trainer': ['MovieSceneSegmentationTrainer'],
        'image_inpainting_trainer': ['ImageInpaintingTrainer'],
        'referring_video_object_segmentation_trainer':
        ['ReferringVideoObjectSegmentationTrainer'],
        'image_defrcn_fewshot_detection_trainer':
        ['ImageDefrcnFewshotTrainer'],
        'cartoon_translation_trainer': ['CartoonTranslationTrainer'],
        'nerf_recon_acc_trainer': ['NeRFReconAccTrainer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
