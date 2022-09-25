# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .audio.ans_trainer import ANSTrainer
    from .base import DummyTrainer
    from .builder import build_trainer
    from .cv import (ImageInstanceSegmentationTrainer,
                     ImagePortraitEnhancementTrainer,
                     MovieSceneSegmentationTrainer)
    from .multi_modal import CLIPTrainer
    from .nlp import SequenceClassificationTrainer, PassageRankingTrainer
    from .nlp_trainer import NlpEpochBasedTrainer, VecoTrainer
    from .trainer import EpochBasedTrainer

else:
    _import_structure = {
        'audio.ans_trainer': ['ANSTrainer'],
        'base': ['DummyTrainer'],
        'builder': ['build_trainer'],
        'cv': [
            'ImageInstanceSegmentationTrainer',
            'ImagePortraitEnhancementTrainer', 'MovieSceneSegmentationTrainer'
        ],
        'multi_modal': ['CLIPTrainer'],
        'nlp': ['SequenceClassificationTrainer', 'PassageRankingTrainer'],
        'nlp_trainer': ['NlpEpochBasedTrainer', 'VecoTrainer'],
        'trainer': ['EpochBasedTrainer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
