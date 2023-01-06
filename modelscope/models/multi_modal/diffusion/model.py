# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os.path as osp
from typing import Any, Dict

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.diffusion.diffusion import (
    GaussianDiffusion, beta_schedule)
from modelscope.models.multi_modal.diffusion.structbert import (BertConfig,
                                                                BertModel)
from modelscope.models.multi_modal.diffusion.tokenizer import FullTokenizer
from modelscope.models.multi_modal.diffusion.unet_generator import \
    DiffusionGenerator
from modelscope.models.multi_modal.diffusion.unet_upsampler_256 import \
    SuperResUNet256
from modelscope.models.multi_modal.diffusion.unet_upsampler_1024 import \
    SuperResUNet1024
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['DiffusionForTextToImageSynthesis']


def make_diffusion(schedule,
                   num_timesteps=1000,
                   init_beta=None,
                   last_beta=None,
                   var_type='fixed_small'):
    betas = beta_schedule(schedule, num_timesteps, init_beta, last_beta)
    diffusion = GaussianDiffusion(betas, var_type=var_type)
    return diffusion


class Tokenizer(object):

    def __init__(self, vocab_file, seq_len=64):
        self.vocab_file = vocab_file
        self.seq_len = seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

    def __call__(self, text):
        # tokenization
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens[:self.seq_len - 2] + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # padding
        input_ids += [0] * (self.seq_len - len(input_ids))
        input_mask += [0] * (self.seq_len - len(input_mask))
        segment_ids += [0] * (self.seq_len - len(segment_ids))
        assert len(input_ids) == len(input_mask) == len(
            segment_ids) == self.seq_len

        # convert to tensors
        input_ids = torch.LongTensor(input_ids)
        input_mask = torch.LongTensor(input_mask)
        segment_ids = torch.LongTensor(segment_ids)
        return input_ids, segment_ids, input_mask


class DiffusionModel(nn.Module):

    def __init__(self, model_dir):
        super(DiffusionModel, self).__init__()
        # including text and generator config
        model_config = json.load(
            open('{}/model_config.json'.format(model_dir), encoding='utf-8'))

        # text encoder
        text_config = model_config['text_config']
        self.text_encoder = BertModel(BertConfig.from_dict(text_config))

        # generator (64x64)
        generator_config = model_config['generator_config']
        self.unet_generator = DiffusionGenerator(**generator_config)

        # upsampler (256x256)
        upsampler_256_config = model_config['upsampler_256_config']
        self.unet_upsampler_256 = SuperResUNet256(**upsampler_256_config)

        # upsampler (1024x1024)
        upsampler_1024_config = model_config['upsampler_1024_config']
        self.unet_upsampler_1024 = SuperResUNet1024(**upsampler_1024_config)

    def forward(self, noise, timesteps, input_ids, token_type_ids,
                attention_mask):
        context, y = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        context = context[-1]
        x = self.unet_generator(noise, timesteps, y, context, attention_mask)
        x = self.unet_upsampler_256(noise, timesteps, x,
                                    torch.zeros_like(timesteps), y, context,
                                    attention_mask)
        x = self.unet_upsampler_1024(x, t, x)
        return x


@MODELS.register_module(
    Tasks.text_to_image_synthesis, module_name=Models.diffusion)
class DiffusionForTextToImageSynthesis(Model):

    def __init__(self, model_dir, device='gpu'):
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        super().__init__(model_dir=model_dir, device=device)
        diffusion_model = DiffusionModel(model_dir=model_dir)
        pretrained_params = torch.load(
            osp.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE), 'cpu')
        diffusion_model.load_state_dict(pretrained_params)
        diffusion_model.eval().to()

        self.device = create_device(device)
        diffusion_model.to(self.device)

        # modules
        self.text_encoder = diffusion_model.text_encoder
        self.unet_generator = diffusion_model.unet_generator
        self.unet_upsampler_256 = diffusion_model.unet_upsampler_256
        self.unet_upsampler_1024 = diffusion_model.unet_upsampler_1024

        # text tokenizer
        vocab_path = f'{model_dir}/{ModelFile.VOCAB_FILE}'
        self.tokenizer = Tokenizer(vocab_file=vocab_path, seq_len=64)

        # diffusion process
        diffusion_params = json.load(
            open(
                '{}/diffusion_config.json'.format(model_dir),
                encoding='utf-8'))
        self.diffusion_generator = make_diffusion(
            **diffusion_params['generator_config'])
        self.diffusion_upsampler_256 = make_diffusion(
            **diffusion_params['upsampler_256_config'])
        self.diffusion_upsampler_1024 = make_diffusion(
            **diffusion_params['upsampler_1024_config'])

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if not all([key in input for key in ('text', 'noise', 'timesteps')]):
            raise ValueError(
                f'input should contains "text", "noise", and "timesteps", but got {input.keys()}'
            )
        input_ids, token_type_ids, attention_mask = self.tokenizer(
            input['text'])
        input_ids = input_ids.to(self.device).unsqueeze(0)
        token_type_ids = token_type_ids.to(self.device).unsqueeze(0)
        attention_mask = attention_mask.to(self.device).unsqueeze(0)
        context, y = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        context = context[-1]
        x = self.unet_generator(noise, timesteps, y, context, attention_mask)
        x = self.unet_upsampler_256(noise, timesteps, x,
                                    torch.zeros_like(timesteps), y, context,
                                    attention_mask)
        x = self.unet_upsampler_1024(x, t, x)
        img = x.clamp(-1, 1).add(1).mul(127.5)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return img

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    @torch.no_grad()
    def generate(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if 'text' not in input:
            raise ValueError(
                f'input should contain "text", but got {input.keys()}')

        # encode text
        input_ids, token_type_ids, attention_mask = self.tokenizer(
            input['text'])
        input_ids = input_ids.to(self.device).unsqueeze(0)
        token_type_ids = token_type_ids.to(self.device).unsqueeze(0)
        attention_mask = attention_mask.to(self.device).unsqueeze(0)
        context, y = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        context = context[-1]

        # choose a proper solver
        solver = input.get('solver', 'dpm-solver')
        if solver == 'dpm-solver':
            # generation
            img = self.diffusion_generator.dpm_solver_sample_loop(
                noise=torch.randn(1, 3, 64, 64).to(self.device),
                model=self.unet_generator,
                model_kwargs=[{
                    'y': y,
                    'context': context,
                    'mask': attention_mask
                }, {
                    'y': torch.zeros_like(y),
                    'context': torch.zeros_like(context),
                    'mask': attention_mask
                }],
                percentile=input.get('generator_percentile', 0.995),
                guide_scale=input.get('generator_guide_scale', 5.0),
                dpm_solver_timesteps=input.get('dpm_solver_timesteps', 20),
                order=3,
                skip_type='logSNR',
                method='singlestep',
                t_start=0.9946)

            # upsampling (64->256)
            if not input.get('debug', False):
                img = F.interpolate(
                    img,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
            img = self.diffusion_upsampler_256.dpm_solver_sample_loop(
                noise=torch.randn_like(img),
                model=self.unet_upsampler_256,
                model_kwargs=[{
                    'lx': img,
                    'lt': torch.zeros(1).to(self.device),
                    'y': y,
                    'context': context,
                    'mask': attention_mask
                }, {
                    'lx': img,
                    'lt': torch.zeros(1).to(self.device),
                    'y': torch.zeros_like(y),
                    'context': torch.zeros_like(context),
                    'mask': torch.zeros_like(attention_mask)
                }],
                percentile=input.get('upsampler_256_percentile', 0.995),
                guide_scale=input.get('upsampler_256_guide_scale', 5.0),
                dpm_solver_timesteps=input.get('dpm_solver_timesteps', 20),
                order=3,
                skip_type='logSNR',
                method='singlestep',
                t_start=0.9946)

            # upsampling (256->1024)
            if not input.get('debug', False):
                img = F.interpolate(
                    img,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
            img = self.diffusion_upsampler_1024.dpm_solver_sample_loop(
                noise=torch.randn_like(img),
                model=self.unet_upsampler_256,
                model_kwargs=[{
                    'lx': img,
                    'lt': torch.zeros(1).to(self.device),
                    'y': y,
                    'context': context,
                    'mask': attention_mask
                }, {
                    'lx': img,
                    'lt': torch.zeros(1).to(self.device),
                    'y': torch.zeros_like(y),
                    'context': torch.zeros_like(context),
                    'mask': torch.zeros_like(attention_mask)
                }],
                percentile=input.get('upsampler_256_percentile', 0.995),
                guide_scale=input.get('upsampler_256_guide_scale', 5.0),
                dpm_solver_timesteps=input.get('dpm_solver_timesteps', 10),
                order=3,
                skip_type='logSNR',
                method='singlestep',
                t_start=None)
        elif solver == 'ddim':
            # generation
            img = self.diffusion_generator.ddim_sample_loop(
                noise=torch.randn(1, 3, 64, 64).to(self.device),
                model=self.unet_generator,
                model_kwargs=[{
                    'y': y,
                    'context': context,
                    'mask': attention_mask
                }, {
                    'y': torch.zeros_like(y),
                    'context': torch.zeros_like(context),
                    'mask': attention_mask
                }],
                percentile=input.get('generator_percentile', 0.995),
                guide_scale=input.get('generator_guide_scale', 5.0),
                ddim_timesteps=input.get('generator_ddim_timesteps', 250),
                eta=input.get('generator_ddim_eta', 0.0))

            # upsampling (64->256)
            if not input.get('debug', False):
                img = F.interpolate(
                    img,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
            img = self.diffusion_upsampler_256.ddim_sample_loop(
                noise=torch.randn_like(img),
                model=self.unet_upsampler_256,
                model_kwargs=[{
                    'lx': img,
                    'lt': torch.zeros(1).to(self.device),
                    'y': y,
                    'context': context,
                    'mask': attention_mask
                }, {
                    'lx': img,
                    'lt': torch.zeros(1).to(self.device),
                    'y': torch.zeros_like(y),
                    'context': torch.zeros_like(context),
                    'mask': torch.zeros_like(attention_mask)
                }],
                percentile=input.get('upsampler_256_percentile', 0.995),
                guide_scale=input.get('upsampler_256_guide_scale', 5.0),
                ddim_timesteps=input.get('upsampler_256_ddim_timesteps', 50),
                eta=input.get('upsampler_256_ddim_eta', 0.0))

            # upsampling (256->1024)
            if not input.get('debug', False):
                img = F.interpolate(
                    img,
                    scale_factor=4.0,
                    mode='bilinear',
                    align_corners=False)
            img = self.diffusion_upsampler_1024.ddim_sample_loop(
                noise=torch.randn_like(img),
                model=self.unet_upsampler_1024,
                model_kwargs={'concat': img},
                percentile=input.get('upsampler_1024_percentile', 0.995),
                ddim_timesteps=input.get('upsampler_1024_ddim_timesteps', 20),
                eta=input.get('upsampler_1024_ddim_eta', 0.0))
        else:
            raise ValueError(
                'currently only supports "ddim" and "dpm-solve" solvers')

        # output
        img = img.clamp(-1, 1).add(1).mul(127.5).squeeze(0).permute(
            1, 2, 0).cpu().numpy().astype(np.uint8)
        return img
