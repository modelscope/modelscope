# Copyright © Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import \
    StableDiffusionPipeline as DiffuserStableDiffusionPipeline
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionPipeline(DiffusersPipeline):

    def __init__(self, model: str, lora_dir: str = None, **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub or local model dir.
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load pipeline
        torch_type = torch.float16 if self.device == 'cuda' else torch.float32
        self.pipeline = DiffuserStableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch_type)
        self.pipeline = self.pipeline.to(self.device)
        # load lora moudle to unet
        if lora_dir is not None:
            assert os.path.exists(lora_dir), f"{lora_dir} isn't exist"
            self.pipeline.unet.load_attn_procs(lora_dir)

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

    def forward(self,
                inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        """
        Args:
            inputs: input dict including text key.
            forward_params['num_inference_steps']: numbers of pipeline steps.
            forward_params['guidance_scale']: guide to classifier free scale guidance.
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )

        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')

        images = self.pipeline(
            inputs['text'],
            num_inference_steps=forward_params['num_inference_steps'],
            guidance_scale=forward_params['guidance_scale'])

        return images

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}
