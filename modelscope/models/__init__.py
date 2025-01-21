# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.automodel_utils import fix_transformers_upgrade
from modelscope.utils.error import (AUDIO_IMPORT_ERROR,
                                    TENSORFLOW_IMPORT_WARNING)
from modelscope.utils.import_utils import (is_torch_available,
                                           is_transformers_available)
from . import audio, cv, multi_modal, nlp
from .base import Head, Model
from .builder import BACKBONES, HEADS, MODELS, build_model

if is_torch_available():
    from .base.base_torch_model import TorchModel
    from .base.base_torch_head import TorchHead

if is_transformers_available():
    fix_transformers_upgrade()
