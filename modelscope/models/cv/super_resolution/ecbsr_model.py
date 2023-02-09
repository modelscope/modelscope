# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .ecb import ECB

logger = get_logger()
__all__ = ['ECBSRModel']


@MODELS.register_module(Tasks.image_super_resolution, module_name=Models.ecbsr)
class ECBSRModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image denoise model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        # network architecture
        self.module_nums = self.config.model.model_args.module_nums
        self.channel_nums = self.config.model.model_args.channel_nums
        self.scale = self.config.model.model_args.scale
        self.colors = self.config.model.model_args.colors
        self.with_idt = self.config.model.model_args.with_idt
        self.act_type = self.config.model.model_args.act_type

        backbone = []
        backbone += [
            ECB(self.colors,
                self.channel_nums,
                depth_multiplier=2.0,
                act_type=self.act_type,
                with_idt=self.with_idt)
        ]
        for i in range(self.module_nums):
            backbone += [
                ECB(self.channel_nums,
                    self.channel_nums,
                    depth_multiplier=2.0,
                    act_type=self.act_type,
                    with_idt=self.with_idt)
            ]
        backbone += [
            ECB(self.channel_nums,
                self.colors * self.scale * self.scale,
                depth_multiplier=2.0,
                act_type='linear',
                with_idt=self.with_idt)
        ]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)

        self.interp = nn.Upsample(scale_factor=self.scale, mode='nearest')

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        output = self.backbone(input)
        output = self.upsampler(output) + self.interp(input)
        return {'outputs': output}

    def forward(self, inputs: Dict[str,
                                   Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """
        return self._inference_forward(**inputs)

    @classmethod
    def _instantiate(cls, **kwargs):
        model_file = kwargs.get('am_model_name', ModelFile.TORCH_MODEL_FILE)
        model_dir = kwargs['model_dir']
        ckpt_path = os.path.join(model_dir, model_file)
        logger.info(f'loading model from {ckpt_path}')
        model_dir = kwargs.pop('model_dir')
        model = cls(model_dir=model_dir, **kwargs)
        ckpt_path = os.path.join(model_dir, model_file)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        return model
