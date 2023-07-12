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

    def __init__(self,
                 model: str,
                 lora_dir: str = None,
                 custom_dir: str = None,
                 modifier_token: str = None,
                 **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub or local model dir.
            lora_dir: lora weight dir for unet.
            custom_dir: custom diffusion weight dir for unet.
            modifier_token: token to use as a modifier for the concept of custom diffusion.
        """
        # check custom diffusion input value
        if custom_dir is None and modifier_token is not None:
            raise ValueError(
                'custom_dir is None but modifier_token is not None')
        elif custom_dir is not None and modifier_token is None:
            raise ValueError(
                'modifier_token is None but custom_dir is not None')

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
        # load custom diffusion to unet
        if custom_dir is not None:
            assert os.path.exists(custom_dir), f"{custom_dir} isn't exist"
            self.pipeline.unet.load_attn_procs(
                custom_dir, weight_name='pytorch_custom_diffusion_weights.bin')
            self.pipeline.load_textual_inversion(
                custom_dir, weight_name=f'{modifier_token}.bin')

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )

        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')

        images = self.pipeline(
            inputs['text'], num_inference_steps=30, guidance_scale=7.5)

        return images

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}
