# Copyright (c) Alibaba, Inc. and its affiliates.
from .base import Model
from .builder import MODELS, build_model

try:
    from .audio.tts.am import SambertNetHifi16k
    from .audio.tts.vocoder import Hifigan16k

except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        pass
    else:
        raise ModuleNotFoundError(e)

try:
    from .audio.kws import GenericKeyWordSpotting
    from .multi_modal import OfaForImageCaptioning
    from .nlp import (BertForMaskedLM, BertForSequenceClassification,
                      CsanmtForTranslation, SbertForNLI,
                      SbertForSentenceSimilarity,
                      SbertForSentimentClassification,
                      SbertForTokenClassification,
                      SbertForZeroShotClassification, SpaceForDialogIntent,
                      SpaceForDialogModeling, SpaceForDialogStateTracking,
                      StructBertForMaskedLM, VecoForMaskedLM)
    from .audio.ans.frcrn import FRCRNModel
except ModuleNotFoundError as e:
    if str(e) == "No module named 'pytorch'":
        pass
    else:
        raise ModuleNotFoundError(e)
