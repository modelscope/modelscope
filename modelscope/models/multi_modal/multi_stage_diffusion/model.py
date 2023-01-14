# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math
import os.path as osp
from typing import Any, Dict

import json
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.multi_stage_diffusion.clip import CLIP
from modelscope.models.multi_modal.multi_stage_diffusion.decoder import Decoder
from modelscope.models.multi_modal.multi_stage_diffusion.gaussian_diffusion import (
    GaussianDiffusion, beta_schedule)
from modelscope.models.multi_modal.multi_stage_diffusion.prior import Prior
from modelscope.models.multi_modal.multi_stage_diffusion.tokenizer import (
    CLIPTokenizer, XGLMTokenizer)
from modelscope.models.multi_modal.multi_stage_diffusion.upsampler import (
    Upsampler256, Upsampler1024)
from modelscope.models.multi_modal.multi_stage_diffusion.xglm import XGLM
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['MultiStageDiffusionForTextToImageSynthesis']


def make_diffusion(schedule,
                   num_timesteps=1000,
                   init_beta=None,
                   last_beta=None,
                   mean_type='eps',
                   var_type='fixed_small'):
    betas = beta_schedule(schedule, num_timesteps, init_beta, last_beta)
    diffusion = GaussianDiffusion(
        betas, mean_type=mean_type, var_type=var_type)
    return diffusion


class UnCLIP(nn.Module):

    def __init__(self, model_dir):
        super(UnCLIP, self).__init__()
        self.model_dir = model_dir
        self.config = json.load(
            open(f'{model_dir}/{ModelFile.CONFIGURATION}', encoding='utf-8'))

        # modules
        self.clip = CLIP(**self.config['clip']).fp16()
        self.xglm = XGLM(**self.config['xglm'])
        self.prior = Prior(**self.config['prior'])
        self.decoder = Decoder(**self.config['decoder'])
        self.upsampler256 = Upsampler256(**self.config['upsampler256'])
        self.upsampler1024 = Upsampler1024(**self.config['upsampler1024'])

        # diffusions
        self.prior_diffusion = make_diffusion(**self.config['prior_diffusion'])
        self.decoder_diffusion = make_diffusion(
            **self.config['decoder_diffusion'])
        self.upsampler256_diffusion = make_diffusion(
            **self.config['upsampler256_diffusion'])
        self.upsampler1024_diffusion = make_diffusion(
            **self.config['upsampler1024_diffusion'])

        # tokenizers
        self.clip_tokenizer = CLIPTokenizer(
            bpe_path=f'{model_dir}/bpe_simple_vocab_16e6.txt.gz')
        self.xglm_tokenizer = XGLMTokenizer(model_dir=model_dir)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            '"forward" is not implemented. Use "synthesis" instead.')

    @torch.no_grad()
    def synthesis(self,
                  text='A photo of a confused grizzly bear in calculus class.',
                  tokenizer='clip',
                  batch_size=4,
                  timesteps_prior=100,
                  timesteps_64=50,
                  timesteps_256=20,
                  timesteps_1024=20,
                  guide_prior=3.0,
                  guide_64=7.0,
                  guide_256=3.0,
                  guide_1024=3.0,
                  eta_prior=0.0,
                  eta_64=0.0,
                  eta_256=0.0,
                  eta_1024=0.0,
                  solver='dpm-solver'):
        device = next(self.parameters()).device

        # check params
        assert all([
            t > 0 and t <= 1000 for t in
            [timesteps_prior, timesteps_64, timesteps_256, timesteps_1024]
        ])
        assert all([
            g > 1 and g < 15
            for g in [guide_prior, guide_64, guide_256, guide_1024]
        ])
        assert all([
            e >= 0 and e <= 1.0
            for e in [eta_prior, eta_64, eta_256, eta_1024]
        ])
        assert batch_size >= 1 and batch_size <= 16

        # tokenize the text
        if tokenizer == 'clip':
            y = F.normalize(
                self.clip.textual(self.clip_tokenizer([text]).to(device)),
                p=2,
                dim=1)
            zero_y = F.normalize(
                self.clip.textual(self.clip_tokenizer(['']).to(device)),
                p=2,
                dim=1)
        elif tokenizer == 'xglm':
            y = F.normalize(
                self.xglm(*to_device(self.xglm_tokenizer([text]), device)),
                p=2,
                dim=1)
            zero_y = F.normalize(
                self.xglm(*to_device(self.xglm_tokenizer(['']), device)),
                p=2,
                dim=1)
        else:
            raise ValueError(
                f'Expected tokenizer to be one of "clip" or "xglm", but got {tokenizer}'
            )
        y = math.sqrt(y.size(1)) * y.repeat(batch_size, 1)
        zero_y = math.sqrt(zero_y.size(1)) * zero_y.repeat(batch_size, 1)

        # synthesis
        with amp.autocast(enabled=True):
            # choose a proper solver
            if solver == 'dpm-solver':
                # prior
                x0 = self.prior_diffusion.dpm_solver_sample_loop(
                    noise=torch.randn_like(y),
                    model=self.prior,
                    model_kwargs=[{
                        'y': y
                    }, {
                        'y': zero_y
                    }],
                    guide_scale=guide_prior,
                    dpm_solver_timesteps=timesteps_prior,
                    order=3,
                    skip_type='logSNR',
                    method='singlestep',
                    t_start=0.9946)

                # decoder
                imgs64 = self.decoder_diffusion.dpm_solver_sample_loop(
                    noise=torch.randn(batch_size, 3, 64, 64).to(device),
                    model=self.decoder,
                    model_kwargs=[{
                        'y': x0
                    }, {
                        'y': torch.zeros_like(x0)
                    }],
                    guide_scale=guide_64,
                    percentile=0.995,
                    dpm_solver_timesteps=timesteps_64,
                    order=3,
                    skip_type='logSNR',
                    method='singlestep',
                    t_start=0.9946).clamp_(-1, 1)

                # upsampler256
                imgs256 = F.interpolate(
                    imgs64,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
                imgs256 = self.upsampler256_diffusion.dpm_solver_sample_loop(
                    noise=torch.randn_like(imgs256),
                    model=self.upsampler256,
                    model_kwargs=[{
                        'y': y,
                        'concat': imgs256
                    }, {
                        'y': zero_y,
                        'concat': imgs256
                    }],
                    guide_scale=guide_256,
                    percentile=0.995,
                    dpm_solver_timesteps=timesteps_256,
                    order=3,
                    skip_type='logSNR',
                    method='singlestep',
                    t_start=0.9946).clamp_(-1, 1)

                # upsampler1024
                imgs1024 = F.interpolate(
                    imgs256,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
                imgs1024 = self.upsampler1024_diffusion.dpm_solver_sample_loop(
                    noise=torch.randn_like(imgs1024),
                    model=self.upsampler1024,
                    model_kwargs=[{
                        'y': y,
                        'concat': imgs1024
                    }, {
                        'y': zero_y,
                        'concat': imgs1024
                    }],
                    guide_scale=guide_1024,
                    percentile=0.995,
                    dpm_solver_timesteps=timesteps_1024,
                    order=3,
                    skip_type='logSNR',
                    method='singlestep',
                    t_start=None).clamp_(-1, 1)
            elif solver == 'ddim':
                # prior
                x0 = self.prior_diffusion.ddim_sample_loop(
                    noise=torch.randn_like(y),
                    model=self.prior,
                    model_kwargs=[{
                        'y': y
                    }, {
                        'y': zero_y
                    }],
                    guide_scale=guide_prior,
                    ddim_timesteps=timesteps_prior,
                    eta=eta_prior)

                # decoder
                imgs64 = self.decoder_diffusion.ddim_sample_loop(
                    noise=torch.randn(batch_size, 3, 64, 64).to(device),
                    model=self.decoder,
                    model_kwargs=[{
                        'y': x0
                    }, {
                        'y': torch.zeros_like(x0)
                    }],
                    guide_scale=guide_64,
                    percentile=0.995,
                    ddim_timesteps=timesteps_64,
                    eta=eta_64).clamp_(-1, 1)

                # upsampler256
                imgs256 = F.interpolate(
                    imgs64,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
                imgs256 = self.upsampler256_diffusion.ddim_sample_loop(
                    noise=torch.randn_like(imgs256),
                    model=self.upsampler256,
                    model_kwargs=[{
                        'y': y,
                        'concat': imgs256
                    }, {
                        'y': zero_y,
                        'concat': imgs256
                    }],
                    guide_scale=guide_256,
                    percentile=0.995,
                    ddim_timesteps=timesteps_256,
                    eta=eta_256).clamp_(-1, 1)

                # upsampler1024
                imgs1024 = F.interpolate(
                    imgs256,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
                imgs1024 = self.upsampler1024_diffusion.ddim_sample_loop(
                    noise=torch.randn_like(imgs1024),
                    model=self.upsampler1024,
                    model_kwargs=[{
                        'y': y,
                        'concat': imgs1024
                    }, {
                        'y': zero_y,
                        'concat': imgs1024
                    }],
                    guide_scale=guide_1024,
                    percentile=0.995,
                    ddim_timesteps=timesteps_1024,
                    eta=eta_1024).clamp_(-1, 1)
            else:
                raise ValueError(
                    'currently only supports "ddim" and "dpm-solve" solvers')

        # output ([B, C, H, W] within range [0, 1])
        imgs1024 = imgs1024.add_(1).mul_(255 / 2.0).permute(0, 2, 3, 1).cpu()
        imgs1024 = [
            Image.fromarray(np.array(u, dtype=np.uint8)) for u in imgs1024
        ]
        return imgs1024


@MODELS.register_module(
    Tasks.text_to_image_synthesis, module_name=Models.multi_stage_diffusion)
class MultiStageDiffusionForTextToImageSynthesis(TorchModel):

    def __init__(self, model_dir, device='gpu'):
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        super().__init__(model_dir=model_dir, device=device)
        model = UnCLIP(model_dir=model_dir)
        pretrained_params = torch.load(
            osp.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE), 'cpu')
        model.load_state_dict(pretrained_params)
        model.eval()

        self.device = create_device(device)
        self.model = model.to(self.device)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(input, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'text' not in input:
            raise ValueError('input should contain "text", but not found')

        # sampling
        imgs = self.model.synthesis(
            text=input.get('text'),
            tokenizer=input.get('tokenizer', 'clip'),
            batch_size=input.get('batch_size', 4),
            timesteps_prior=input.get('timesteps_prior', 100),
            timesteps_64=input.get('timesteps_64', 50),
            timesteps_256=input.get('timesteps_256', 20),
            timesteps_1024=input.get('timesteps_1024', 20),
            guide_prior=input.get('guide_prior', 3.0),
            guide_64=input.get('guide_64', 7.0),
            guide_256=input.get('guide_256', 3.0),
            guide_1024=input.get('guide_1024', 3.0),
            eta_prior=input.get('eta_prior', 0.0),
            eta_64=input.get('eta_64', 0.0),
            eta_256=input.get('eta_256', 0.0),
            eta_1024=input.get('eta_1024', 0.0),
            solver=input.get('solver', 'dpm-solver'))
        imgs = [np.array(u)[..., ::-1] for u in imgs]
        return imgs
