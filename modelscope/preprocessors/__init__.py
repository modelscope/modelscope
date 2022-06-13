# Copyright (c) Alibaba, Inc. and its affiliates.

from .base import Preprocessor
from .builder import PREPROCESSORS, build_preprocessor
from .common import Compose
from .image import LoadImage, load_image
from .nlp import *  # noqa F403
from .nlp import TextGenerationPreprocessor
from .space.dialog_intent_preprocessor import *  # noqa F403
