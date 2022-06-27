# Copyright (c) Alibaba, Inc. and its affiliates.

from .audio.kws import GenericKeyWordSpotting
from .audio.tts.am import SambertNetHifi16k
from .audio.tts.vocoder import Hifigan16k
from .base import Model
from .builder import MODELS, build_model
from .multi_modal import OfaForImageCaptioning
from .nlp import (BertForSequenceClassification, SbertForNLI,
                  SbertForSentenceSimilarity, SbertForSentimentClassification,
                  SbertForTokenClassification, StructBertForMaskedLM,
                  VecoForMaskedLM)
