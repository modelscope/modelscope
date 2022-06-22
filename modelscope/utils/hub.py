# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from modelscope.hub.constants import MODEL_ID_SEPARATOR
from modelscope.hub.utils.utils import get_cache_dir


# temp solution before the hub-cache is in place
def get_model_cache_dir(model_id: str):
    return os.path.join(get_cache_dir(), model_id)
