# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.utils.error import (AUDIO_IMPORT_ERROR,
                                    TENSORFLOW_IMPORT_WARNING)
from .base import Model
from .builder import MODELS, build_model

try:
    from .audio.tts import SambertHifigan
    from .audio.kws import GenericKeyWordSpotting
    from .audio.ans.frcrn import FRCRNModel
except ModuleNotFoundError as e:
    print(AUDIO_IMPORT_ERROR.format(e))

try:
    from .nlp.csanmt_for_translation import CsanmtForTranslation
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        print(TENSORFLOW_IMPORT_WARNING.format('CsanmtForTranslation'))
    else:
        raise ModuleNotFoundError(e)

try:
    from .multi_modal import OfaForImageCaptioning
    from .nlp import (BertForMaskedLM, BertForSequenceClassification,
                      SbertForNLI, SbertForSentenceSimilarity,
                      SbertForSentimentClassification,
                      SbertForTokenClassification,
                      SbertForZeroShotClassification, SpaceForDialogIntent,
                      SpaceForDialogModeling, SpaceForDialogStateTracking,
                      StructBertForMaskedLM, VecoForMaskedLM)
except ModuleNotFoundError as e:
    if str(e) == "No module named 'pytorch'":
        pass
    else:
        raise ModuleNotFoundError(e)
