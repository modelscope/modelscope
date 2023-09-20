# Copyright 2023-2024 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.checkpoint import save_checkpoint, save_configuration
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    Tasks.text_to_image_synthesis, module_name=Models.stable_diffusion)
class StableDiffusion(TorchModel):
    """ The implementation of stable diffusion model based on TorchModel.

    This model is constructed with the implementation of stable diffusion model. If you want to
    finetune lightweight parameters on your own dataset, you can define you own tuner module
    and load in this cls.
    """

    def __init__(self, model_dir, *args, **kwargs):
        """ Initialize a vision stable diffusion model.

        Args:
          model_dir: model id or path
        """
        super().__init__(model_dir, *args, **kwargs)
        revision = kwargs.pop('revision', None)
        xformers_enable = kwargs.pop('xformers_enable', False)
        self.lora_tune = kwargs.pop('lora_tune', False)
        self.dreambooth_tune = kwargs.pop('dreambooth_tune', False)

        self.weight_dtype = kwargs.pop('torch_type', torch.float32)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load scheduler, tokenizer and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_dir, subfolder='scheduler')
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_dir, subfolder='tokenizer', revision=revision)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_dir, subfolder='text_encoder', revision=revision)
        self.vae = AutoencoderKL.from_pretrained(
            model_dir, subfolder='vae', revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_dir, subfolder='unet', revision=revision)
        self.safety_checker = None

        # Freeze gradient calculation and move to device
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder = self.text_encoder.to(
                self.device, dtype=self.weight_dtype)
        if self.unet is not None:
            if self.lora_tune:
                self.unet.requires_grad_(False)
            self.unet = self.unet.to(self.device, dtype=self.weight_dtype)

        # xformers accelerate memory efficient attention
        if xformers_enable:
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warn(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. '
                    'If you observe problems during training, please update xFormers to at least 0.0.17.'
                )
            self.unet.enable_xformers_memory_efficient_attention()

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

    def forward(self, text='', target=None):
        self.unet.train()
        self.unet = self.unet.to(self.device)

        # Convert to latent space
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
        noisy_latents = self.noise_scheduler.add_noise(latents, noise,
                                                       timesteps)

        input_ids = self.tokenize_caption(text).to(self.device)

        # Get the text embedding for conditioning
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise,
                                                       timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.noise_scheduler.config.prediction_type}'
            )

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps,
                               encoder_hidden_states).sample

        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')

        output = {OutputKeys.LOSS: loss}
        return output

    def save_pretrained(self,
                        target_folder: Union[str, os.PathLike],
                        save_checkpoint_names: Union[str, List[str]] = None,
                        save_function: Callable = partial(
                            save_checkpoint, with_meta=False),
                        config: Optional[dict] = None,
                        save_config_function: Callable = save_configuration,
                        **kwargs):
        # Skip copying the original weights for lora and dreambooth method
        if self.lora_tune or self.dreambooth_tune:
            config['pipeline']['type'] = 'diffusers-stable-diffusion'
            pass
        else:
            super().save_pretrained(target_folder, save_checkpoint_names,
                                    save_function, config,
                                    save_config_function, **kwargs)
