# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import List, Union

from numpy import deprecate

from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.utils.utils import get_cache_dir
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile


# temp solution before the hub-cache is in place
@deprecate
def get_model_cache_dir(model_id: str):
    return os.path.join(get_cache_dir(), model_id)


def read_config(model_id_or_path: str):
    """ Read config from hub or local path

    Args:
        model_id_or_path (str): Model repo name or local directory path.

    Return:
        config (:obj:`Config`): config object
    """
    if not os.path.exists(model_id_or_path):
        local_path = model_file_download(model_id_or_path,
                                         ModelFile.CONFIGURATION)
    else:
        local_path = os.path.join(model_id_or_path, ModelFile.CONFIGURATION)

    return Config.from_file(local_path)


def auto_load(model: Union[str, List[str]]):
    if isinstance(model, str):
        if not osp.exists(model):
            model = snapshot_download(model)
    else:
        model = [
            snapshot_download(m) if not osp.exists(m) else m for m in model
        ]

    return model
