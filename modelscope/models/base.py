# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.builder import build_model
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()

Tensor = Union['torch.Tensor', 'tf.Tensor']


class Model(ABC):

    def __init__(self, model_dir, *args, **kwargs):
        self.model_dir = model_dir

    def __call__(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.postprocess(self.forward(input))

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass

    def postprocess(self, input: Dict[str, Tensor],
                    **kwargs) -> Dict[str, Tensor]:
        """ Model specific postprocess and convert model output to
        standard model outputs.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        return input

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        revision: Optional[str] = 'master',
                        *model_args,
                        **kwargs):
        """ Instantiate a model from local directory or remote model repo. Note
        that when loading from remote, the model revision can be specified.
        """
        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        else:
            local_model_dir = snapshot_download(model_name_or_path, revision)
        logger.info(f'initialize model from {local_model_dir}')
        cfg = Config.from_file(
            osp.join(local_model_dir, ModelFile.CONFIGURATION))
        task_name = cfg.task
        model_cfg = cfg.model
        assert hasattr(
            cfg, 'pipeline'), 'pipeline config is missing from config file.'
        pipeline_cfg = cfg.pipeline
        # TODO @wenmeng.zwm may should manually initialize model after model building
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type

        model_cfg.model_dir = local_model_dir

        for k, v in kwargs.items():
            model_cfg[k] = v
        model = build_model(model_cfg, task_name)

        # dynamically add pipeline info to model for pipeline inference
        model.pipeline = pipeline_cfg
        return model
