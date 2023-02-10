# Copyright © Alibaba, Inc. and its affiliates.

import math
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.preprocessors.image import load_image
from modelscope.utils.config import Config
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.image_inpainting, module_name=Pipelines.image_inpainting_sdv2)
class ImageInpaintingSDV2Pipeline(DiffusersPipeline):
    """ Stable Diffusion for Image Inpainting Pipeline.

    Example:

    >>> import cv2
    >>> from modelscope.outputs import OutputKeys
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> input_location = 'data/test/images/image_inpainting/image_inpainting.png'
    >>> input_mask_location = 'data/test/images/image_inpainting/image_inpainting_mask.png'
    >>> prompt = 'background'

    >>> input = {
    >>>     'image': input_location,
    >>>     'mask': input_mask_location,
    >>>     'prompt': prompt
    >>> }
    >>> image_inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_stable-diffusion-v2_image-inpainting_base')
    >>> output = image_inpainting(input)[OutputKeys.OUTPUT_IMG]
    >>> cv2.imwrite('result.png', output)

    """

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        """
        Use `model` to create a stable diffusion pipeline for image inpainting.
        Args:
            model: model id on modelscope hub.
            device: str = 'gpu'
        """
        super().__init__(model, device, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float16)

        # build upon the diffuser stable diffusion pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model, torch_dtype=torch_dtype)
        self.pipeline.to(self.device)

        enable_attention_slicing = kwargs.get('enable_attention_slicing', True)
        if enable_attention_slicing:
            self.pipeline.enable_attention_slicing()

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method

        Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = pipeline_parameters
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, pipeline_parameters

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )

        num_inference_steps = inputs.get('num_inference_steps', 50)
        guidance_scale = inputs.get('guidance_scale', 7.5)
        negative_prompt = inputs.get('negative_prompt', None)
        num_images_per_prompt = inputs.get('num_images_per_prompt', 1)
        eta = inputs.get('eta', 0.0)

        if 'prompt' in inputs.keys():
            prompt = inputs['prompt']
        else:
            # for demo_service
            prompt = forward_params.get('prompt', 'background')
        print(f'Test with prompt: {prompt}')

        image = load_image(inputs['image'])
        mask = load_image(inputs['mask'])

        w, h = image.size
        print(f'loaded input image of size ({w}, {h})')
        width, height = map(lambda x: x - x % 64,
                            (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        mask = mask.resize((width, height))
        out_image = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta).images[0]

        return {'result': out_image}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = np.array(inputs['result'])
        return {OutputKeys.OUTPUT_IMG: result[:, :, ::-1]}
