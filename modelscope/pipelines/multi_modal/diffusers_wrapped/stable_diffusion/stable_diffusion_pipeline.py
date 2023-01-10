# Copyright © Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch
from diffusers import StableDiffusionPipeline

from modelscope.metainfo import Pipelines
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks


# Wrap around the diffusers stable diffusion pipeline implementation
# for a unified ModelScope pipeline experience. Native stable diffusion
# pipelines will be implemented in later releases.
@PIPELINES.register_module(
    Tasks.diffusers_stable_diffusion,
    module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionWrapperPipeline(DiffusersPipeline):

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub.
            device: str = 'gpu'
        """
        super().__init__(model, device, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float16)

        # build upon the diffuser stable diffusion pipeline
        self.diffusers_pipeline = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch_dtype)
        self.diffusers_pipeline.to(self.device)

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, prompt, **kwargs):
        return self.diffusers_pipeline(prompt, **kwargs)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
