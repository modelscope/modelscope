# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:

    from .clip import CLIPForMultiModalEmbedding
    from .clip_interrogator import CLIP_Interrogator
    from .diffusion import DiffusionForTextToImageSynthesis
    from .efficient_diffusion_tuning import EfficientStableDiffusion
    from .gemm import GEMMForMultiModalEmbedding
    from .mmr import VideoCLIPForMultiModalEmbedding
    from .mplug_for_all_tasks import HiTeAForAllTasks, MPlugForAllTasks
    from .mplug_owl import MplugOwlForConditionalGeneration
    from .multi_stage_diffusion import \
        MultiStageDiffusionForTextToImageSynthesis
    from .ofa_for_all_tasks import OfaForAllTasks
    from .ofa_for_text_to_image_synthesis_model import \
        OfaForTextToImageSynthesis
    from .prost import ProSTForTVRetrieval
    from .rleg import RLEGForMultiModalEmbedding
    from .team import TEAMForMultiModalSimilarity
    from .video_synthesis import TextToVideoSynthesis
    from .vldoc import VLDocForDocVLEmbedding
    from .videocomposer import VideoComposer

else:
    _import_structure = {
        'clip': ['CLIPForMultiModalEmbedding'],
        'diffusion': ['DiffusionForTextToImageSynthesis'],
        'gemm': ['GEMMForMultiModalEmbedding'],
        'rleg': ['RLEGForMultiModalEmbedding'],
        'team': ['TEAMForMultiModalSimilarity'],
        'mmr': ['VideoCLIPForMultiModalEmbedding'],
        'prost': ['ProSTForTVRetrieval'],
        'mplug_for_all_tasks': ['MPlugForAllTasks', 'HiTeAForAllTasks'],
        'ofa_for_all_tasks': ['OfaForAllTasks'],
        'ofa_for_text_to_image_synthesis_model':
        ['OfaForTextToImageSynthesis'],
        'multi_stage_diffusion':
        ['MultiStageDiffusionForTextToImageSynthesis'],
        'vldoc': ['VLDocForDocVLEmbedding'],
        'video_synthesis': ['TextToVideoSynthesis'],
        'efficient_diffusion_tuning': ['EfficientStableDiffusion'],
        'mplug_owl': ['MplugOwlForConditionalGeneration'],
        'clip_interrogator': ['CLIP_Interrogator'],
        'videocomposer': ['VideoComposer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
