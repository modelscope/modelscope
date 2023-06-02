# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.data import Dataset
import hashlib
import random
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union

from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel, DiffusionPipeline)
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from modelscope.metainfo import Trainers
from modelscope.outputs import OutputKeys, ModelOutputBase
from modelscope.models.base import TorchModel
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import CheckpointProcessor
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.torch_utils import is_dist
from modelscope.utils.constant import ModeKeys
from modelscope.trainers.trainer import EpochBasedTrainer

# class StableDiffusionModel(TorchModel):
#     def __init__(self, noise_scheduler, tokenizer, vae, text_encoder, unet, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.noise_scheduler = noise_scheduler
#         self.tokenizer = tokenizer
#         self.vae = vae
#         self.text_encoder = text_encoder
#         self.unet = unet

#     def forward(self, *args, **kwargs):
#         return self.model.forward(*args, **kwargs)


@TRAINERS.register_module(module_name=Trainers.stable_diffusion)
class StableDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        self.revision = kwargs.pop("revision", None)
        self.pretrained_model_name_or_path = kwargs.pop('pretrained_model_name_or_path', 
                                                        "runwayml/stable-diffusion-v1-5")
        super().__init__(*args, **kwargs)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder='scheduler').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='tokenizer',
            revision=self.revision).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='vae',
            revision=self.revision).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='unet',
            revision=self.revision).to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='text_encoder',
            revision=self.revision).to(self.device)

        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.unet is not None:
            self.unet.requires_grad_(False)
    
    def train_step(self, model, inputs):
        """ Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`TorchModel`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # EvaluationHook will do evaluate and change mode to val, return to train mode
        # TODO: find more pretty way to change mode
        model.train()
        self._mode = ModeKeys.TRAIN

        image = inputs["target"]
        prompt = inputs["prompt"]
        num_batches = image.shape[0]

        image = image.to(self.dtype)
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True)['input_ids'].to(self.device)

        encoder_hidden_states = self.text_encoder(input_ids)[0]

        if self.scheduler.config.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError('Unknown prediction type '
                                f'{self.scheduler.config.prediction_type}')

        # NOTE: we train unet in fp32, convert to float manually
        model_output = self.unet(
            noisy_latents.float(),
            timesteps,
            encoder_hidden_states=encoder_hidden_states.float())
        model_pred = model_output['sample']

        # calculate loss in FP32
        loss = F.mse_loss(model_pred.float(), gt.float())

        train_outputs = {OutputKeys.LOSS: loss}

        if isinstance(train_outputs, ModelOutputBase):
            train_outputs = train_outputs.to_dict()
        if not isinstance(train_outputs, dict):
            raise TypeError('"model.forward()" must return a dict')

        # add model output info to log
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update(
                    [key for key in train_outputs.keys() if key_p in key])

            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if is_dist():
                        value = value.data.clone().to('cuda')
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])

        self.train_outputs = train_outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self,
                        state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)

    def data_assemble(self, inputs):
        batch = {}
        batch["pixel_values"] = inputs["target"]
        text_inputs = self.tokenize_prompt(self.tokenizer, self.instance_prompt)
        batch["input_ids"] = text_inputs.input_ids
        batch["attention_mask"] = text_inputs.attention_mask

        return batch