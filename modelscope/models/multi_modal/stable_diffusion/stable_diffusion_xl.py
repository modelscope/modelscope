# Copyright 2023-2024 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import random
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import (AutoTokenizer, CLIPTextModel,
                          CLIPTextModelWithProjection)

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.checkpoint import save_checkpoint, save_configuration
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    Tasks.text_to_image_synthesis, module_name=Models.stable_diffusion_xl)
class StableDiffusionXL(TorchModel):
    """ The implementation of stable diffusion xl model based on TorchModel.

    This model is constructed with the implementation of stable diffusion xl model. If you want to
    finetune lightweight parameters on your own dataset, you can define you own tuner module
    and load in this cls.
    """

    def __init__(self, model_dir, *args, **kwargs):
        """ Initialize a vision stable diffusion xl model.

        Args:
          model_dir: model id or path
        """
        super().__init__(model_dir, *args, **kwargs)
        revision = kwargs.pop('revision', None)
        xformers_enable = kwargs.pop('xformers_enable', False)
        self.lora_tune = kwargs.pop('lora_tune', False)
        self.resolution = kwargs.pop('resolution', 1024)
        self.random_flip = kwargs.pop('random_flip', True)

        self.weight_dtype = torch.float32
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load scheduler, tokenizer and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_dir, subfolder='scheduler')
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            model_dir,
            subfolder='tokenizer',
            revision=revision,
            use_fast=False)
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            model_dir,
            subfolder='tokenizer_2',
            revision=revision,
            use_fast=False)
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            model_dir, subfolder='text_encoder', revision=revision)
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_dir, subfolder='text_encoder_2', revision=revision)
        self.vae = AutoencoderKL.from_pretrained(
            model_dir, subfolder='vae', revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_dir, subfolder='unet', revision=revision)
        self.safety_checker = None

        # Freeze gradient calculation and move to device
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae = self.vae.to(self.device)
        if self.text_encoder_one is not None:
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_one = self.text_encoder_one.to(self.device)
        if self.text_encoder_two is not None:
            self.text_encoder_two.requires_grad_(False)
            self.text_encoder_two = self.text_encoder_two.to(self.device)
        if self.unet is not None:
            if self.lora_tune:
                self.unet.requires_grad_(False)
            self.unet = self.unet.to(self.device)

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

    def tokenize_caption(self, tokenizer, captions):
        """ Convert caption text to token data.

        Args:
            tokenizer: the tokenizer one or two.
            captions: a batch of texts.
        Returns: token's data as tensor.
        """
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return inputs.input_ids

    def compute_time_ids(self, original_size, crops_coords_top_left):
        target_size = (self.resolution, self.resolution)
        add_time_ids = list(original_size + crops_coords_top_left
                            + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.device, dtype=self.weight_dtype)
        return add_time_ids

    def encode_prompt(self,
                      text_encoders,
                      tokenizers,
                      prompt,
                      text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def preprocessing_data(self, text, target):
        train_crop = transforms.RandomCrop(self.resolution)
        train_resize = transforms.Resize(
            self.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        image = target
        original_size = (image.size()[-1], image.size()[-2])
        image = train_resize(image)
        y1, x1, h, w = train_crop.get_params(
            image, (self.resolution, self.resolution))
        image = crop(image, y1, x1, h, w)
        if self.random_flip and random.random() < 0.5:
            # flip
            x1 = image.size()[-2] - x1
            image = train_flip(image)
        crop_top_left = (y1, x1)
        input_ids_one = self.tokenize_caption(self.tokenizer_one, text)
        input_ids_two = self.tokenize_caption(self.tokenizer_two, text)

        return original_size, crop_top_left, image, input_ids_one, input_ids_two

    def forward(self, text='', target=None):
        self.unet.train()
        self.unet = self.unet.to(self.device)

        # processing data
        original_size, crop_top_left, image, input_ids_one, input_ids_two = self.preprocessing_data(
            text, target)
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

        add_time_ids = self.compute_time_ids(original_size, crop_top_left)

        # Predict the noise residual
        unet_added_conditions = {'time_ids': add_time_ids}
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            text_encoders=[self.text_encoder_one, self.text_encoder_two],
            tokenizers=None,
            prompt=None,
            text_input_ids_list=[input_ids_one, input_ids_two])
        unet_added_conditions.update({'text_embeds': pooled_prompt_embeds})
        # Predict the noise residual and compute loss
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(model_input, noise,
                                                       timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.noise_scheduler.config.prediction_type}'
            )

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
        if self.lora_tune:
            config['pipeline']['type'] = 'diffusers-stable-diffusion-xl'
            pass
        else:
            super().save_pretrained(target_folder, save_checkpoint_names,
                                    save_function, config,
                                    save_config_function, **kwargs)
