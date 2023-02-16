# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.import_utils import is_torch_available
from .base_head import *  # noqa F403
from .base_model import *  # noqa F403

if is_torch_available():
    from .base_torch_model import TorchModel
    from .base_torch_head import TorchHead
