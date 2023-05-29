# Copyright © Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import cv2
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import (StableDiffusionPipeline, AutoencoderKL, 
                       DDPMScheduler, UNet2DConditionModel)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks


# Wrap around the diffusers stable diffusion pipeline implementation
# for a unified ModelScope pipeline experience. Native stable diffusion
# pipelines will be implemented in later releases.
@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionWrapperPipeline(DiffusersPipeline):

    def __init__(self, model: str, unet_folder: str = None, text_encoder_folder: str = None, device: str = 'gpu', **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub.
            device: str = 'gpu'
        """
        super().__init__(model, device, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        # build complete the diffuser stable diffusion pipeline
        if unet_folder is None and text_encoder_folder is None:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model, torch_dtype=torch_dtype)
        # build respectively diffuser stable diffusion pipeline
        else:
            if unet_folder is not None:
                unet = UNet2DConditionModel.from_pretrained(unet_folder)
            else:
                unet = UNet2DConditionModel.from_pretrained(model, subfolder='unet')
            if text_encoder_folder is not None:
                text_encoder = CLIPTextModel.from_pretrained(text_encoder_folder)
            else:
                text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(model, subfolder='vae')
            tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
            scheduler = DDPMScheduler.from_pretrained(model, subfolder='scheduler')
            self.pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
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
