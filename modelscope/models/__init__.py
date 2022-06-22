# Copyright (c) Alibaba, Inc. and its affiliates.

# from .audio.tts.am import SambertNetHifi16k
# from .audio.tts.vocoder import Hifigan16k
from .base import Model
from .builder import MODELS, build_model
# from .multi_model import OfaForImageCaptioning
from .nlp import (
    BertForSequenceClassification,
    SbertForNLI,
    SbertForSentenceSimilarity,
    SbertForSentimentClassification,
    SbertForZeroShotClassification,
    StructBertForMaskedLM,
    VecoForMaskedLM,
    SbertForTokenClassification,
)
