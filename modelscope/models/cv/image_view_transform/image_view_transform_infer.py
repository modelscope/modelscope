# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math
import os
import sys
import time
from contextlib import nullcontext
from functools import partial

import cv2
import diffusers  # 0.12.1
import fire
import numpy as np
import rich
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms

from modelscope.fileio import load
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .ldm.ddim import DDIMSampler
from .util import instantiate_from_config, load_and_preprocess

logger = get_logger()


def load_model_from_config(model, config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


@MODELS.register_module(
    Tasks.image_view_transform, module_name=Models.image_view_transform)
class ImageViewTransform(TorchModel):
    """initialize the image view translation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
    """

    def __init__(self, model_dir, device='cpu', *args, **kwargs):

        super().__init__(model_dir=model_dir, device=device, *args, **kwargs)

        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        config = os.path.join(model_dir,
                              'sd-objaverse-finetune-c_concat-256.yaml')
        ckpt = os.path.join(model_dir, 'zero123-xl.ckpt')
        config = OmegaConf.load(config)
        self.model = None
        self.model = load_model_from_config(
            self.model, config, ckpt, device=self.device)

    def forward(self, model_path, x, y):
        pred_results = _infer(self.model, model_path, x, y, self.device)
        return pred_results


def infer(genmodel, model_path, image_path, target_view_path, device):
    output_ims = genmodel(model_path, image_path, target_view_path)
    return output_ims


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps,
                 n_samples, scale, ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([
                math.radians(x),
                math.sin(math.radians(y)),
                math.cos(math.radians(y)), z
            ])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [
                model.encode_first_stage(
                    (input_im.to(c.device))).mode().detach().repeat(
                        n_samples, 1, 1, 1)
            ]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [
                    torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)
                ]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp(
                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(models, input_im, preprocess, carvekit_path):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)

    if preprocess:

        # model_carvekit = create_carvekit_interface()
        model_carvekit = torch.load(carvekit_path)
        input_im = load_and_preprocess(model_carvekit, input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    return input_im


def main_run(models,
             device,
             return_what,
             x=0.0,
             y=0.0,
             z=0.0,
             raw_im=None,
             carvekit_path=None,
             preprocess=True,
             scale=3.0,
             n_samples=4,
             ddim_steps=50,
             ddim_eta=1.0,
             precision='fp32',
             h=256,
             w=256):
    '''
    :param raw_im (PIL Image).
    '''

    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    input_im = preprocess_image(models, raw_im, preprocess, carvekit_path)

    if 'gen' in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(models)
        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(input_im, models, sampler, precision, h,
                                      w, ddim_steps, n_samples, scale,
                                      ddim_eta, used_x, y, z)

        output_ims = []
        for x_sample in x_samples_ddim:
            image = x_sample.detach().cpu().squeeze().numpy()
            image = np.transpose(image, (1, 2, 0)) * 255
            image = np.uint8(image)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            output_ims.append(bgr)

        return output_ims


def _infer(genmodel, model_path, image_path, target_view_path, device):
    if isinstance(image_path, str):
        raw_image = load(image_path)
        print(type(raw_image))
    else:
        raw_image = image_path
    if isinstance(target_view_path, str):
        views = load(target_view_path)
    else:
        views = target_view_path
    # views = views.astype(np.float32)
    carvekit_path = os.path.join(model_path, 'carvekit.pth')
    output_ims = main_run(genmodel, device, 'angles_gen', views[0], views[1],
                          views[2], raw_image, carvekit_path, views[3],
                          views[4], views[5], views[6], views[7])
    return output_ims
