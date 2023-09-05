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
from diffusers.models import attention
from diffusers.utils import deprecation_utils
from swift import AdapterConfig, LoRAConfig, PromptConfig, Swift
from transformers import CLIPTextModel, CLIPTokenizer

from modelscope import snapshot_download
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.efficient_diffusion_tuning.sd_lora import \
    LoRATuner
from modelscope.outputs import OutputKeys
from modelscope.utils.checkpoint import save_checkpoint, save_configuration
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .control_sd_lora import ControlLoRATuner

utils.deprecate = lambda *arg, **kwargs: None
deprecation_utils.deprecate = lambda *arg, **kwargs: None
attention.deprecate = lambda *arg, **kwargs: None

__tuner_MAP__ = {'lora': LoRATuner, 'control_lora': ControlLoRATuner}


@MODELS.register_module(
    Tasks.efficient_diffusion_tuning,
    module_name=Models.efficient_diffusion_tuning)
class EfficientStableDiffusion(TorchModel):
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
        tuner_name = kwargs.pop('tuner_name', 'lora')
        pretrained_model_name_or_path = kwargs.pop(
            'pretrained_model_name_or_path',
            'AI-ModelScope/stable-diffusion-v1-5')
        pretrained_model_name_or_path = snapshot_download(
            pretrained_model_name_or_path)
        tuner_config = kwargs.pop('tuner_config', None)
        pretrained_tuner = kwargs.pop('pretrained_tuner', None)
        revision = kwargs.pop('revision', None)
        inference = kwargs.pop('inference', True)

        if pretrained_tuner is not None:
            pretrained_tuner = osp.join(model_dir, pretrained_tuner)

        self.weight_dtype = torch.float32
        self.inference = inference

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if self.inference:
            self.pipe = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                torch_dtype=self.weight_dtype,
                safety_checker=None)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            self.unet = self.pipe.unet
        else:
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
            self.unet.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
        self.is_control = tuner_name.startswith('control_')
        self.tuner_name = tuner_name

        if tuner_name == 'swift-lora':
            rank = tuner_config[
                'rank'] if tuner_config and 'rank' in tuner_config else 4
            lora_config = LoRAConfig(
                r=rank,
                target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
                merge_weights=False,
                use_merged_linear=False)
            self.unet = Swift.prepare_model(self.unet, lora_config)
        elif tuner_name == 'swift-adapter':
            adapter_length = tuner_config[
                'adapter_length'] if tuner_config and 'adapter_length' in tuner_config else 10
            adapter_config = AdapterConfig(
                dim=-1,
                hidden_pos=0,
                target_modules=r'.*ff\.net\.2$',
                adapter_length=adapter_length)
            self.unet = Swift.prepare_model(self.unet, adapter_config)
        elif tuner_name == 'swift-prompt':
            prompt_length = tuner_config[
                'prompt_length'] if tuner_config and 'prompt_length' in tuner_config else 10
            prompt_config = PromptConfig(
                dim=[
                    320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 640, 640,
                    640, 320, 320, 320
                ],
                target_modules=
                r'.*[down_blocks|up_blocks|mid_block]\.\d+\.attentions\.\d+\.transformer_blocks\.\d+$',
                embedding_pos=0,
                prompt_length=prompt_length,
                attach_front=False)
            self.unet = Swift.prepare_model(self.unet, prompt_config)
        elif tuner_name in ('lora', 'control_lora'):
            # if not set the config of control-tuner, we add the lora tuner directly to the original framework,
            # otherwise the control side network is also added.
            tuner_cls = __tuner_MAP__[tuner_name]
            tuner = tuner_cls.tune(
                self,
                tuner_config=osp.join(model_dir, tuner_config),
                pretrained_tuner=pretrained_tuner)
            self.tuner = tuner

    def train(self, mode: bool = True):
        self.training = mode
        if hasattr(self, 'tuner'):
            self.tuner.train(mode=mode)
        else:
            super().train(mode=mode)

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True):
        if hasattr(self, 'tuner'):
            self.tuner.load_state_dict(state_dict=state_dict, strict=strict)
        else:
            super().load_state_dict(state_dict=state_dict, strict=strict)

    def state_dict(self):
        if hasattr(self, 'tuner'):
            return self.tuner.state_dict()
        elif self.tuner_name.startswith('swift'):
            return self.unet.state_dict()
        else:
            return super().state_dict()

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

    def forward(self, prompt='', cond=None, target=None, **args):
        if self.inference:
            if 'generator_seed' in args and isinstance(args['generator_seed'],
                                                       int):
                generator = torch.Generator(device=self.device).manual_seed(
                    args['generator_seed'])
            else:
                generator = None
            num_inference_steps = args.get('num_inference_steps', 30)
            if self.is_control:
                _ = self.tuner(cond.to(self.device)).control_states
            images = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator).images
            return images
        else:
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

            # Inject control states to unet
            if self.is_control:
                _ = self.tuner(cond.to(dtype=self.weight_dtype)).control_states
            # else:
            #     tune_weights_list = self.tuner()

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

    def parameters(self, recurse: bool = True):
        if hasattr(self, 'tuner'):
            return self.tuner.parameters(recurse=recurse)
        else:
            return super().parameters(recurse=recurse)

    def save_pretrained(self,
                        target_folder: Union[str, os.PathLike],
                        save_checkpoint_names: Union[str, List[str]] = None,
                        save_function: Callable = partial(
                            save_checkpoint, with_meta=False),
                        config: Optional[dict] = None,
                        save_config_function: Callable = save_configuration,
                        **kwargs):

        if config is None and hasattr(self, 'cfg'):
            config = self.cfg

        config['model']['inference'] = True
        config['model']['pretrained_tuner'] = 'pytorch_model.bin'
        super().save_pretrained(target_folder, save_checkpoint_names,
                                save_function, config, save_config_function,
                                **kwargs)

    @classmethod
    def _instantiate(cls, model_dir, **kwargs):
        config = Config.from_file(osp.join(model_dir, ModelFile.CONFIGURATION))
        for k, v in kwargs.items():
            config.model[k] = v

        model = EfficientStableDiffusion(
            model_dir,
            pretrained_model_name_or_path=config.model.
            pretrained_model_name_or_path,
            tuner_name=config.model.tuner_name,
            tuner_config=config.model.tuner_config,
            pretrained_tuner=config.model.get('pretrained_tuner', None),
            inference=config.model.get('inference', False))
        model.config = config
        return model
