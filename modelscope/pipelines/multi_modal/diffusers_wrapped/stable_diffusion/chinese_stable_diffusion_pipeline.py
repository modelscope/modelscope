# Copyright 2022 The HuggingFace Team.
# Copyright 2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

# The implementation here is modified based on diffusers,
# originally Apache License, Copyright 2022 The HuggingFace Team,
# and publicly available at
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from PIL import Image
from transformers import (ChineseCLIPProcessor, ChineseCLIPTextModel,
                          CLIPFeatureExtractor)

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.chinese_stable_diffusion)
class ChineseStableDiffusionPipeline(DiffusersPipeline):

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub.
            device: str = 'gpu'
        """
        super().__init__(model, device, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        self.pipeline = _DiffuersChineseStableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch_dtype)
        self.pipeline.text_encoder.pooler = None
        self.pipeline.to(self.device)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')

        return self.pipeline(
            prompt=inputs.get('text'),
            height=inputs.get('height'),
            width=inputs.get('width'),
            num_inference_steps=inputs.get('num_inference_steps', 50),
            guidance_scale=inputs.get('guidance_scale', 7.5),
            negative_prompt=inputs.get('negative_prompt'),
            num_images_per_prompt=inputs.get('num_images_per_prompt', 1),
            eta=inputs.get('eta', 0.0),
            generator=inputs.get('generator'),
            latents=inputs.get('latents'),
            output_type=inputs.get('output_type', 'pil'),
            return_dict=inputs.get('return_dict', True),
            callback=inputs.get('callback'),
            callback_steps=inputs.get('callback_steps', 1))

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}


class _DiffuersChineseStableDiffusionPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Chinese Stable Diffusion.

    This model inherits from [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`ChineseCLIPTextModel`]):
            Frozen text-encoder. Chinese Stable Diffusion uses the text portion of [ChineseCLIP]
            (https://huggingface.co/docs/transformers/main/en/model_doc/chinese_clip#transformers.ChineseCLIPTextModel),
            specifically the [chinese-clip-vit-huge-patch14]
            (https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14) variant.
        tokenizer (`ChineseCLIPProcessor`):
            Tokenizer of class
            [ChineseCLIPProcessor](https://huggingface.co/docs/transformers/main/en/model_doc/chinese_clip#transformers.ChineseCLIPProcessor).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ['safety_checker', 'feature_extractor']

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ChineseCLIPTextModel,
        tokenizer: ChineseCLIPProcessor,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
                         EulerDiscreteScheduler,
                         EulerAncestralDiscreteScheduler,
                         DPMSolverMultistepScheduler, ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker)

    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                text=prompt,
                padding='max_length',
                truncation=True,
                max_length=52,
                return_tensors='pt')
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            prompt_embeds = self.text_encoder(**text_inputs)
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(
            dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
                                           seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !='
                    f' {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.')
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                text=uncond_tokens,
                padding='max_length',
                truncation=True,
                max_length=52,
                return_tensors='pt')
            uncond_input = {k: v.to(device) for k, v in uncond_input.items()}
            negative_prompt_embeds = self.text_encoder(**uncond_input)
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
