# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import List, Optional, Union

from requests import HTTPError

from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from .logger import get_logger

logger = get_logger(__name__)


def create_model_if_not_exist(
        api,
        model_id: str,
        chinese_name: str,
        visibility: Optional[int] = ModelVisibility.PUBLIC,
        license: Optional[str] = Licenses.APACHE_V2,
        revision: Optional[str] = 'master'):
    exists = True
    try:
        api.get_model(model_id=model_id, revision=revision)
    except HTTPError:
        exists = False
    if exists:
        print(f'model {model_id} already exists, skip creation.')
        return False
    else:
        api.create_model(
            model_id=model_id,
            visibility=visibility,
            license=license,
            chinese_name=chinese_name,
        )
        print(f'model {model_id} successfully created.')
        return True


def read_config(model_id_or_path: str, revision: Optional[str] = 'master'):
    """ Read config from hub or local path

    Args:
        model_id_or_path (str): Model repo name or local directory path.

    Return:
        config (:obj:`Config`): config object
    """
    if not os.path.exists(model_id_or_path):
        local_path = model_file_download(
            model_id_or_path, ModelFile.CONFIGURATION, revision=revision)
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


def get_model_type(model_dir):
    try:
        configuration_file = osp.join(model_dir, ModelFile.CONFIGURATION)
        config_file = osp.join(model_dir, 'config.json')
        if osp.isfile(configuration_file):
            cfg = Config.from_file(configuration_file)
            return cfg.model.model_type if hasattr(cfg.model, 'model_type') and not hasattr(cfg.model, 'type') \
                else cfg.model.type
        elif osp.isfile(config_file):
            cfg = Config.from_file(config_file)
            return cfg.model_type if hasattr(cfg, 'model_type') else None
    except Exception as e:
        logger.error(f'parse config file failed with error: {e}')
