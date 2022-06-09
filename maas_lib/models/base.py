# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from maas_hub.file_download import model_file_download
from maas_hub.snapshot_download import snapshot_download

from maas_lib.models.builder import build_model
from maas_lib.utils.config import Config
from maas_lib.utils.constant import CONFIGFILE
from maas_lib.utils.hub import get_model_cache_dir

Tensor = Union['torch.Tensor', 'tf.Tensor']


class Model(ABC):

    def __init__(self, model_dir, *args, **kwargs):
        self.model_dir = model_dir

    def __call__(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.post_process(self.forward(input))

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass

    def post_process(self, input: Dict[str, Tensor],
                     **kwargs) -> Dict[str, Tensor]:
        # model specific postprocess, implementation is optional
        # will be called in Pipeline and evaluation loop(in the future)
        return input

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *model_args, **kwargs):
        """ Instantiate a model from local directory or remote model repo
        """
        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        else:
            cache_path = get_model_cache_dir(model_name_or_path)
            local_model_dir = cache_path if osp.exists(
                cache_path) else snapshot_download(model_name_or_path)
            # else:
            #     raise ValueError(
            #         'Remote model repo {model_name_or_path} does not exists')

        cfg = Config.from_file(osp.join(local_model_dir, CONFIGFILE))
        task_name = cfg.task
        model_cfg = cfg.model
        # TODO @wenmeng.zwm may should manually initialize model after model building
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type
        model_cfg.model_dir = local_model_dir
        return build_model(model_cfg, task_name)
