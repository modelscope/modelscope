# Part of the implementation is borrowed and modified from ControlNet,
# publicly available at https://github.com/lllyasviel/ControlNet

import math
import os
import random
import sys
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from control_ldm.cldm.hack import disable_verbosity, enable_sliced_attention
from control_ldm.cldm.model import create_model, load_state_dict
from control_ldm.ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.compatible_with_transformers import \
    compatible_position_ids
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

__all__ = ['ControlNet']


@MODELS.register_module(
    Tasks.controllable_image_generation,
    module_name=Models.controllable_image_generation)
class ControlNet(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize ControlNet from theq `model_dir` path.
        ControlNet:
            Adding Conditional Control to Text-to-Image Diffusion Models.
            Paper: https://arxiv.org/abs/2302.05543
            Origin codes: https://github.com/lllyasviel/ControlNet
        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        enable_sa = self.config.modelsetting.get('enable_sliced_attention',
                                                 True)
        self.image_resolution = self.config.modelsetting.image_resolution

        self.modelsetting = self.config.modelsetting

        disable_verbosity()
        if enable_sa:
            enable_sliced_attention()
        init_control_type = kwargs.get('control_type', 'hed')
        if init_control_type == 'scribble':
            input_setting = self.modelsetting.scribble
        elif init_control_type == 'canny':
            input_setting = self.modelsetting.canny
        elif init_control_type == 'hough':
            input_setting = self.modelsetting.hough
        elif init_control_type == 'hed':
            input_setting = self.modelsetting.hed
        elif init_control_type == 'depth':
            input_setting = self.modelsetting.depth
        elif init_control_type == 'normal':
            input_setting = self.modelsetting.normal
        elif init_control_type == 'pose':
            input_setting = self.modelsetting.pose
        elif init_control_type == 'seg':
            input_setting = self.modelsetting.seg
        elif init_control_type == 'fake_scribble':
            input_setting = self.modelsetting.scribble
        else:
            print('Error input type, use HED for default!')
            input_setting = self.modelsetting.hed
        self.init_control_type = init_control_type
        self.input_setting = input_setting

        yaml_path = os.path.join(self.model_dir, input_setting.yaml_path)
        ckpt_path = os.path.join(self.model_dir, input_setting.ckpt_path)
        device = kwargs.get('device', 'cuda')
        if device == 'gpu':
            device = 'cuda'
        model = create_model(yaml_path).cpu()
        state_dict = load_state_dict(ckpt_path, location=device)
        compatible_position_ids(
            state_dict,
            'cond_stage_model.transformer.text_model.embeddings.position_ids')
        model.load_state_dict(state_dict)
        self.model = model.to(device)
        self.ddim_sampler = DDIMSampler(self.model)

    def get_resolution(self):
        return self.image_resolution

    def get_config(self):
        return self.modelsetting

    def get_model_dir(self):
        return self.model_dir

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """return the result by the model
        Args:
            inputs (Dict[str, Any]) should contains the keys:
            - "image", a numpy array
            - "prompt", string of prompt
            - "detected_map", a numpy array of detected map
            - "save_memory", boolean indicating whether to save memory
            - "is_cat_img", boolean indicating whether to concatenate results

            inputs (Dict[str, Any]) can also contains the keys, but not required:
            - "image_resolution", int
            - "strength", float
            - "guess_mode", bool
            - "ddim_steps", int
            - "scale", float
            - "num_samples", int
            - "eta", float
            - "a_prompt", string of added prompt
            - "n_prompt", string of negative prompt

        Returns:
            Dict[str, Any]: A dict contains result, detected_map and boolean 'is_cat_img'
            indicating whether to concatenate the result and the detected_map.

        """
        image = inputs['image']
        prompt = inputs['prompt']
        detected_map = inputs['detected_map']  # processed in preprocessor
        save_memory = inputs.get('save_memory', False)

        num_samples = inputs.get('num_samples', self.input_setting.num_samples)
        scale = inputs.get('scale', self.input_setting.scale)
        ddim_steps = inputs.get('ddim_steps', self.input_setting.ddim_steps)
        eta = inputs.get('eta', self.input_setting.eta)
        a_prompt = inputs.get('a_prompt', self.input_setting.a_prompt)
        n_prompt = inputs.get('n_prompt', self.input_setting.n_prompt)
        guess_mode = inputs.get('guess_mode', self.input_setting.guess_mode)
        strength = inputs.get('strength', self.input_setting.strength)
        print(f'Process with guess_mode:{guess_mode},strength:{strength},')
        print(
            f'num_samples:{num_samples},scale:{scale},ddim_steps:{ddim_steps},eta:{eta}'
        )
        print(f'a_prompt:\'{a_prompt}\',n_prompt:\'{n_prompt}\',')

        with torch.no_grad():
            H, W, C = image.shape

            control = torch.from_numpy(
                detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {
                'c_concat': [control],
                'c_crossattn': [
                    self.model.get_learned_conditioning(
                        [prompt + ', ' + a_prompt] * num_samples)
                ]
            }
            un_cond = {
                'c_concat':
                [torch.zeros_like(control) if guess_mode else control],
                'c_crossattn': [
                    self.model.get_learned_conditioning([n_prompt]
                                                        * num_samples)
                ]
            }
            shape = (4, H // 8, W // 8)

            if save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [
                strength * (0.825**float(12 - i)) for i in range(13)
            ] if guess_mode else ([strength] * 13)
            samples, intermediates = self.ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond)

            if save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5
                + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]

        if self.init_control_type == 'hough':
            show_det_map = cv2.dilate(
                detected_map,
                np.ones(shape=(3, 3), dtype=np.uint8),
                iterations=1)
        elif self.init_control_type == 'normal':
            show_det_map = detected_map[:, :, ::-1]
        elif self.init_control_type == 'fake_scribble' or self.init_control_type == 'scribble':
            show_det_map = 255 - detected_map
        else:
            show_det_map = detected_map
        return {
            'result': results,
            'detected_map': show_det_map,
            'is_cat_img': inputs['is_cat_img']
        }
