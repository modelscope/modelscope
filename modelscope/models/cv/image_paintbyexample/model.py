# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf
from paint_ldm.util import instantiate_from_config

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

LOGGER = get_logger()


def load_model_from_config(config, ckpt, verbose=False):
    LOGGER.info(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        LOGGER.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        LOGGER.info('missing keys:')
        LOGGER.info(m)
    if len(u) > 0 and verbose:
        LOGGER.info('unexpected keys:')
        LOGGER.info(u)

    return model


@MODELS.register_module(
    Tasks.image_paintbyexample, module_name=Models.image_paintbyexample)
class StablediffusionPaintbyexample(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, **kwargs)

        config = OmegaConf.load(os.path.join(model_dir, 'v1.yaml'))
        model = load_model_from_config(
            config, os.path.join(model_dir, 'pytorch_model.pt'))
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)
