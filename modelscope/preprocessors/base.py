# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModeKeys, Tasks
from modelscope.utils.hub import read_config, snapshot_download
from modelscope.utils.logger import get_logger
from .builder import build_preprocessor

logger = get_logger(__name__)


class Preprocessor(ABC):

    def __init__(self, mode=ModeKeys.INFERENCE, *args, **kwargs):
        self._mode = mode
        self.device = int(
            os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else None
        pass

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        cfg_dict: Config = None,
                        preprocessor_mode=ModeKeys.INFERENCE,
                        **kwargs):
        """ Instantiate a model from local directory or remote model repo. Note
        that when loading from remote, the model revision can be specified.
        """
        if not os.path.exists(model_name_or_path):
            model_dir = snapshot_download(
                model_name_or_path, revision=revision)
        else:
            model_dir = model_name_or_path
        if cfg_dict is None:
            cfg = read_config(model_dir)
        else:
            cfg = cfg_dict
        task = cfg.task
        if 'task' in kwargs:
            task = kwargs.pop('task')
        field_name = Tasks.find_field_by_task(task)
        if not hasattr(cfg, 'preprocessor'):
            logger.error('No preprocessor field found in cfg.')
            return None

        sub_key = 'train' if preprocessor_mode == ModeKeys.TRAIN else 'val'

        if 'type' not in cfg.preprocessor:
            if sub_key in cfg.preprocessor:
                sub_cfg = getattr(cfg.preprocessor, sub_key)
            else:
                logger.error(
                    f'No {sub_key} key and type key found in '
                    f'preprocessor domain of configuration.json file.')
                return None
        else:
            sub_cfg = cfg.preprocessor

        if len(sub_cfg):
            if isinstance(sub_cfg, Sequence):
                # TODO: for Sequence, need adapt to `mode` and `mode_dir` args,
                # and add mode for Compose or other plans
                raise NotImplementedError('Not supported yet!')
            sub_cfg = deepcopy(sub_cfg)
            sub_cfg.update({'model_dir': model_dir})
            sub_cfg.update(kwargs)
            preprocessor = build_preprocessor(sub_cfg, field_name)
        else:
            logger.error(
                f'Cannot find available config to build preprocessor at mode {preprocessor_mode}, '
                f'please check the preprocessor field in the configuration.json file.'
            )
            return None
        preprocessor.mode = preprocessor_mode
        return preprocessor
