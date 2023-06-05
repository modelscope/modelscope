# Copyright 2023-2024 The Alibaba Fundamental Vision Team Authors. All rights reserved.
# The implementation is adopted from HighCWu,
# made pubicly available under the Apache License 2.0 License at https://github.com/HighCWu/ControlLoRA
import os
import os.path as osp
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, UNet2DConditionModel,
                       utils)
from diffusers.models import cross_attention
from diffusers.utils import deprecation_utils
from transformers import CLIPTextModel, CLIPTokenizer

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.tuners.control_sd_lora import ControlLoRATuner
from modelscope.tuners.sd_lora import LoRATuner
from modelscope.utils.checkpoint import save_checkpoint, save_configuration
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Models.stable_diffusion)
class StableDiffusion(TorchModel):
    """ The implementation of efficient diffusion tuning model based on TorchModel.

    This model is constructed with the implementation of stable diffusion model. If you want to
    finetune lightweight parameters on your own dataset, you can define you own tuner module
    and load in this cls.
    """

    def __init__(self, model_dir, *args, **kwargs):
        """ Initialize a vision efficient diffusion tuning model.

        Args:
          model_dir: model id or path, where model_dir/pytorch_model.bin
        """
        super().__init__(model_dir, *args, **kwargs)
        pretrained_model_name_or_path = kwargs.pop(
            'pretrained_model_name_or_path', 'runwayml/stable-diffusion-v1-5')
        revision = kwargs.pop('revision', None)

        self.weight_dtype = torch.float32
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder='scheduler')
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder='tokenizer',
            revision=revision)
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder='text_encoder',
            revision=revision)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder='vae',
            revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder='unet',
            revision=revision)
        
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.unet is not None:
            self.unet.requires_grad_(False)

    # def train(self, mode: bool = True):
    #     self.training = mode
    #     if hasattr(self, 'tuner'):
    #         self.tuner.train(mode=mode)

    # def load_state_dict(self,
    #                     state_dict: Mapping[str, Any],
    #                     strict: bool = True):
    #     if hasattr(self, 'tuner'):
    #         self.tuner.load_state_dict(state_dict=state_dict, strict=strict)
    #     else:
    #         return super().load_state_dict(
    #             state_dict=state_dict, strict=strict)

    # def state_dict(self):
    #     if hasattr(self, 'tuner'):
    #         return self.tuner.state_dict()
    #     else:
    #         return super().state_dict()

    def tokenize_caption(self, captions):
        """ Convert caption text to token data.

        Args:
          captions: a batch of texts.
        Returns: token's data as tensor.
        """
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return inputs.input_ids

    def forward(self, prompt='', target=None):
        with torch.no_grad():
            latents = self.vae.encode(
                target.to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps, (bsz, ),
            device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)

        input_ids = self.tokenize_caption(prompt).to(self.device)

        # Get the text embedding for conditioning
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(
                latents, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.noise_scheduler.config.prediction_type}'
            )

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps,
                                encoder_hidden_states).sample
        loss = F.mse_loss(
            model_pred.float(), target.float(), reduction='mean')
        output = {OutputKeys.LOSS: loss}
        return output

    # def parameters(self, recurse: bool = True):
    #     if hasattr(self, 'tuner'):
    #         return self.tuner.parameters(recurse=recurse)
    #     else:
    #         return super().parameters(recurse=recurse)

    # def save_pretrained(self,
    #                     target_folder: Union[str, os.PathLike],
    #                     save_checkpoint_names: Union[str, List[str]] = None,
    #                     save_function: Callable = partial(
    #                         save_checkpoint, with_meta=False),
    #                     config: Optional[dict] = None,
    #                     save_config_function: Callable = save_configuration,
    #                     **kwargs):

    #     if config is None and hasattr(self, 'cfg'):
    #         config = self.cfg

    #     config['model']['inference'] = True
    #     super().save_pretrained(target_folder, save_checkpoint_names,
    #                             save_function, config, save_config_function,
    #                             **kwargs)

    # @classmethod
    # def _instantiate(cls, model_dir, **kwargs):
    #     config = Config.from_file(osp.join(model_dir, ModelFile.CONFIGURATION))
    #     for k, v in kwargs.items():
    #         config.model[k] = v

    #     model = EfficientStableDiffusion(
    #         model_dir,
    #         pretrained_model_name_or_path=config.model.
    #         pretrained_model_name_or_path,
    #         tuner_name=config.model.tuner_name,
    #         tuner_config=config.model.tuner_config,
    #         pretrained_tuner=config.model.get('pretrained_tuner', None),
    #         inference=config.model.get('inference', False))
    #     model.config = config
    #     return model
