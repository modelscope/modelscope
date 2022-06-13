# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
from typing import List, Union

import json
from maas_hub.file_download import model_file_download
from matplotlib.pyplot import get

from modelscope.utils.config import Config
from modelscope.utils.constant import CONFIGFILE
from modelscope.utils.logger import get_logger

logger = get_logger()


def is_config_has_model(cfg_file):
    try:
        cfg = Config.from_file(cfg_file)
        return hasattr(cfg, 'model')
    except Exception as e:
        logger.error(f'parse config file {cfg_file} failed: {e}')
        return False


def is_model_name(model: Union[str, List]):
    """ whether model is a valid modelhub path
    """

    def is_model_name_impl(model):
        if osp.exists(model):
            cfg_file = osp.join(model, CONFIGFILE)
            if osp.exists(cfg_file):
                return is_config_has_model(cfg_file)
            else:
                return False
        else:
            try:
                cfg_file = model_file_download(model, CONFIGFILE)
                return is_config_has_model(cfg_file)
            except Exception:
                return False

    if isinstance(model, str):
        return is_model_name_impl(model)
    else:
        results = [is_model_name_impl(m) for m in model]
        all_true = all(results)
        any_true = any(results)
        if any_true and not all_true:
            raise ValueError('some model are hub address, some are not')

        return all_true
