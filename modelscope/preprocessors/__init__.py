# Copyright (c) Alibaba, Inc. and its affiliates.

from .base import Preprocessor
from .builder import PREPROCESSORS, build_preprocessor
from .common import Compose
from .image import LoadImage, load_image
from .kws import WavToLists
from .text_to_speech import *  # noqa F403

try:
    from .audio import LinearAECAndFbank
    from .multi_modal import *  # noqa F403
    from .nlp import *  # noqa F403
    from .space.dialog_intent_prediction_preprocessor import *  # noqa F403
    from .space.dialog_modeling_preprocessor import *  # noqa F403
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        pass
    else:
        raise ModuleNotFoundError(e)
