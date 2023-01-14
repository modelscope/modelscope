# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import List, Optional, Union

import torch

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()


def is_config_has_model(cfg_file):
    try:
        cfg = Config.from_file(cfg_file)
        return hasattr(cfg, 'model')
    except Exception as e:
        logger.error(f'parse config file {cfg_file} failed: {e}')
        return False


def is_official_hub_path(path: Union[str, List],
                         revision: Optional[str] = DEFAULT_MODEL_REVISION):
    """ Whether path is an official hub name or a valid local
    path to official hub directory.
    """

    def is_official_hub_impl(path):
        if osp.exists(path):
            cfg_file = osp.join(path, ModelFile.CONFIGURATION)
            return osp.exists(cfg_file)
        else:
            try:
                _ = HubApi().get_model(path, revision=revision)
                return True
            except Exception as e:
                raise ValueError(f'invalid model repo path {e}')

    if isinstance(path, str):
        return is_official_hub_impl(path)
    else:
        results = [is_official_hub_impl(m) for m in path]
        all_true = all(results)
        any_true = any(results)
        if any_true and not all_true:
            raise ValueError(
                f'some model are hub address, some are not, model list: {path}'
            )

        return all_true


def is_model(path: Union[str, List]):
    """ whether path is a valid modelhub path and containing model config
    """

    def is_modelhub_path_impl(path):
        if osp.exists(path):
            cfg_file = osp.join(path, ModelFile.CONFIGURATION)
            if osp.exists(cfg_file):
                return is_config_has_model(cfg_file)
            else:
                return False
        else:
            try:
                cfg_file = model_file_download(path, ModelFile.CONFIGURATION)
                return is_config_has_model(cfg_file)
            except Exception:
                return False

    if isinstance(path, str):
        return is_modelhub_path_impl(path)
    else:
        results = [is_modelhub_path_impl(m) for m in path]
        all_true = all(results)
        any_true = any(results)
        if any_true and not all_true:
            raise ValueError(
                f'some models are hub address, some are not, model list: {path}'
            )

        return all_true


def batch_process(model, data):
    if model.__class__.__name__ == 'OfaForAllTasks':
        # collate batch data due to the nested data structure
        assert isinstance(data, list)
        batch_data = {
            'nsentences': len(data),
            'samples': [d['samples'][0] for d in data],
            'net_input': {}
        }
        for k in data[0]['net_input'].keys():
            batch_data['net_input'][k] = torch.cat(
                [d['net_input'][k] for d in data])
        if 'w_resize_ratios' in data[0]:
            batch_data['w_resize_ratios'] = torch.cat(
                [d['w_resize_ratios'] for d in data])
        if 'h_resize_ratios' in data[0]:
            batch_data['h_resize_ratios'] = torch.cat(
                [d['h_resize_ratios'] for d in data])

        return batch_data
