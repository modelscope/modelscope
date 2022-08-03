# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.builder import build_model
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.hub import parse_label_mapping
from modelscope.utils.logger import get_logger

logger = get_logger()

Tensor = Union['torch.Tensor', 'tf.Tensor']


class Model(ABC):

    def __init__(self, model_dir, *args, **kwargs):
        self.model_dir = model_dir
        device_name = kwargs.get('device', 'gpu')
        assert device_name in ['gpu',
                               'cpu'], 'device should be either cpu or gpu.'
        self._device_name = device_name

    def __call__(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.postprocess(self.forward(input))

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Run the forward pass for a model.

        Args:
            input (Dict[str, Tensor]): the dict of the model inputs for the forward method

        Returns:
            Dict[str, Tensor]: output from the model forward pass
        """
        pass

    def postprocess(self, input: Dict[str, Tensor],
                    **kwargs) -> Dict[str, Tensor]:
        """ Model specific postprocess and convert model output to
        standard model outputs.

        Args:
            input:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return input

    @classmethod
    def _instantiate(cls, **kwargs):
        """ Define the instantiation method of a model,default method is by
            calling the constructor. Note that in the case of no loading model
            process in constructor of a task model, a load_model method is
            added, and thus this method is overloaded
        """
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        cfg_dict: Config = None,
                        *model_args,
                        **kwargs):
        """ Instantiate a model from local directory or remote model repo. Note
        that when loading from remote, the model revision can be specified.
        """
        prefetched = kwargs.get('model_prefetched')
        if prefetched is not None:
            kwargs.pop('model_prefetched')

        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        else:
            if prefetched is True:
                raise RuntimeError(
                    'Expecting model is pre-fetched locally, but is not found.'
                )
            local_model_dir = snapshot_download(model_name_or_path, revision)
        logger.info(f'initialize model from {local_model_dir}')
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg = Config.from_file(
                osp.join(local_model_dir, ModelFile.CONFIGURATION))
        task_name = cfg.task
        model_cfg = cfg.model
        # TODO @wenmeng.zwm may should manually initialize model after model building

        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type

        model_cfg.model_dir = local_model_dir
        for k, v in kwargs.items():
            model_cfg[k] = v
        model = build_model(
            model_cfg, task_name=task_name, default_args=kwargs)

        # dynamically add pipeline info to model for pipeline inference
        if hasattr(cfg, 'pipeline'):
            model.pipeline = cfg.pipeline
        return model
