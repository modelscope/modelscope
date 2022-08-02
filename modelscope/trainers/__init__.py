from .base import DummyTrainer
from .builder import build_trainer
from .cv import (ImageInstanceSegmentationTrainer,
                 ImagePortraitEnhancementTrainer)
from .multi_modal import CLIPTrainer
from .nlp import SequenceClassificationTrainer
from .trainer import EpochBasedTrainer
