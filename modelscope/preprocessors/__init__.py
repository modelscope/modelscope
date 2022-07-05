# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.error import AUDIO_IMPORT_ERROR, TENSORFLOW_IMPORT_ERROR
from .base import Preprocessor
from .builder import PREPROCESSORS, build_preprocessor
from .common import Compose
from .image import LoadImage, load_image
from .kws import WavToLists
from .text_to_speech import *  # noqa F403

try:
    from .audio import LinearAECAndFbank
except ModuleNotFoundError as e:
    print(AUDIO_IMPORT_ERROR.format(e))

try:
    from .multi_modal import *  # noqa F403
    from .nlp import *  # noqa F403
    from .space.dialog_intent_prediction_preprocessor import *  # noqa F403
    from .space.dialog_modeling_preprocessor import *  # noqa F403
    from .space.dialog_state_tracking_preprocessor import *  # noqa F403
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        print(TENSORFLOW_IMPORT_ERROR.format('tts'))
    else:
        raise ModuleNotFoundError(e)
