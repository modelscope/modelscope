# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
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
from .common import ResidualBlockNoBN, make_layer

logger = get_logger()
__all__ = ['MSRResNetLiteModel']


@MODELS.register_module(
    Tasks.video_super_resolution, module_name=Models.msrresnet_lite)
class MSRResNetLiteModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video super-resolution model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        self.max_seq_len = 1

        # network architecture
        in_nc = self.config.model.model_args.in_nc
        out_nc = self.config.model.model_args.out_nc
        nf = self.config.model.model_args.nf
        nb = self.config.model.model_args.nb
        self.upscale = self.config.model.model_args.upscale

        self.conv_first = nn.Conv2d(in_nc, nf // 2, 3, 1, 1, bias=True)
        # use stride=2 conv to downsample
        self.conv_down = nn.Conv2d(nf // 2, nf, 3, 2, 1, bias=True)
        self.recon_trunk = make_layer(ResidualBlockNoBN, nb, mid_channels=nf)

        # upsampling
        if self.upscale == 2:
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(nf // 4, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf // 4, out_nc, 3, 1, 1, bias=True)
        elif self.upscale == 1:
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.conv_last = nn.Conv2d(nf // 4, out_nc, 3, 1, 1, bias=True)
        elif self.upscale == 4:
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.upconv1 = nn.Conv2d(nf // 4, nf, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf // 4, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf // 4, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        if input.ndim == 5:
            input = input.squeeze(1)

        fea = self.lrelu(self.conv_first(input))
        fea = self.lrelu(self.conv_down(fea))
        out = self.recon_trunk(fea)

        out = self.lrelu(self.pixel_shuffle(out))

        if self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(
                input,
                scale_factor=self.upscale,
                mode='bilinear',
                align_corners=False)
            out += base
        elif self.upscale == 1:
            out = self.conv_last(out) + input
        elif self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(
                input,
                scale_factor=self.upscale,
                mode='bilinear',
                align_corners=False)
            out += base

        output = torch.clamp(out, 0.0, 1.0)

        if output.ndim == 4:
            output = output.unsqueeze(1)
        return {'output': output}

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
