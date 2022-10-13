# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule, is_torch_available

if TYPE_CHECKING:
    from .base import TaskDataset
    from .builder import TASK_DATASETS, build_task_dataset
    from .torch_base_dataset import TorchTaskDataset
    from .veco_dataset import VecoDataset
    from .image_instance_segmentation_coco_dataset import ImageInstanceSegmentationCocoDataset
    from .movie_scene_segmentation import MovieSceneSegmentationDataset
    from .video_summarization_dataset import VideoSummarizationDataset
    from .image_inpainting import ImageInpaintingDataset
    from .passage_ranking_dataset import PassageRankingDataset

else:
    _import_structure = {
        'base': ['TaskDataset'],
        'builder': ['TASK_DATASETS', 'build_task_dataset'],
        'torch_base_dataset': ['TorchTaskDataset'],
        'passage_ranking_dataset': ['PassageRankingDataset'],
        'veco_dataset': ['VecoDataset'],
        'image_instance_segmentation_coco_dataset':
        ['ImageInstanceSegmentationCocoDataset'],
        'video_summarization_dataset': ['VideoSummarizationDataset'],
        'movie_scene_segmentation': ['MovieSceneSegmentationDataset'],
        'image_inpainting': ['ImageInpaintingDataset'],
        'sidd_image_denoising_dataset': ['SiddImageDenoisingDataset'],
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
