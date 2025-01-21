# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .audio import ANSTrainer, KanttsTrainer
    from .base import DummyTrainer
    from .builder import build_trainer
    from .cv import (ImageInstanceSegmentationTrainer,
                     ImagePortraitEnhancementTrainer,
                     MovieSceneSegmentationTrainer, ImageInpaintingTrainer,
                     ReferringVideoObjectSegmentationTrainer)
    from .multi_modal import CLIPTrainer
    from .nlp import SequenceClassificationTrainer, TextRankingTrainer, SiameseUIETrainer
    from .nlp_trainer import NlpEpochBasedTrainer, VecoTrainer
    from .trainer import EpochBasedTrainer
    from .training_args import TrainingArgs, build_dataset_from_file
    from .hooks import Hook, Priority

else:
    _import_structure = {
        'audio': ['ANSTrainer', 'KanttsTrainer'],
        'base': ['DummyTrainer'],
        'builder': ['build_trainer'],
        'cv': [
            'ImageInstanceSegmentationTrainer',
            'ImagePortraitEnhancementTrainer', 'MovieSceneSegmentationTrainer',
            'ImageInpaintingTrainer'
        ],
        'multi_modal': ['CLIPTrainer'],
        'nlp': [
            'SequenceClassificationTrainer', 'TextRankingTrainer',
            'SiameseUIETrainer'
        ],
        'nlp_trainer': ['NlpEpochBasedTrainer', 'VecoTrainer'],
        'trainer': ['EpochBasedTrainer'],
        'training_args': ['TrainingArgs', 'build_dataset_from_file'],
        'hooks': ['Hook', 'Priority']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
