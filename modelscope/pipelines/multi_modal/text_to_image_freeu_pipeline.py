# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal.freeu import (
    register_free_crossattn_upblock2d, register_free_upblock2d)
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['FreeUTextToImagePipeline']


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.freeu_stable_diffusion_text2image)
class FreeUTextToImagePipeline(Pipeline):

    def __init__(self, model=str, preprocessor=None, **kwargs):
        """  FreeU Text to Image Pipeline.

        Examples:

        >>> import cv2
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.utils.constant import Tasks

        >>> prompt = "a photo of a running corgi"  # prompt
        >>> output_image_path = './result.png'
        >>> inputs = {'prompt': prompt}
        >>>
        >>> pipe = pipeline(
        >>>     Tasks.text_to_image_synthesis,
        >>>     model='damo/multi-modal_freeu_stable_diffusion',
        >>>     base_model='AI-ModelScope/stable-diffusion-v1-5',
        >>> )
        >>>
        >>> output = pipe(inputs)['output_imgs']
        >>> cv2.imwrite(output_image_path, output)
        >>> print('pipeline: the output image path is {}'.format(output_image_path))
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        self._device = getattr(
            kwargs, 'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        base_model = kwargs.get(
            'base_model', 'AI-ModelScope/stable-diffusion-v1-5')  # default 1.5
        self.freeu_params = kwargs.get('freeu_params', {
            'b1': 1.5,
            'b2': 1.6,
            's1': 0.9,
            's2': 0.2
        })  # default

        logger.info('load freeu stable diffusion text to image pipeline done')
        self.pipeline = pipeline(
            task=Tasks.text_to_image_synthesis,
            model=base_model,
            torch_dtype=torch_dtype,
            device=self._device).pipeline

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Inputs Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(inputs)}'
            )
        # -------- freeu block registration
        register_free_upblock2d(self.pipeline, **self.freeu_params)
        register_free_crossattn_upblock2d(self.pipeline, **self.freeu_params)
        # -------- freeu block registration

        output = self.pipeline(
            prompt=inputs.get('prompt'),
            height=inputs.get('height'),
            width=inputs.get('width'),
            num_inference_steps=inputs.get('num_inference_steps', 50),
            guidance_scale=inputs.get('guidance_scale', 7.5),
            negative_prompt=inputs.get('negative_prompt'),
            num_images_per_prompt=inputs.get('num_images_per_prompt', 1),
            eta=inputs.get('eta', 0.0),
            generator=inputs.get('generator'),
            latents=inputs.get('latents'),
        ).images[0]

        return {'output_tensor': output}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output_img = np.array(inputs['output_tensor'])
        return {OutputKeys.OUTPUT_IMGS: output_img[:, :, ::-1]}
