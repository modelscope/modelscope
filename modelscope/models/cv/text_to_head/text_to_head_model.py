# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.utils import load_image

from modelscope.models import MODELS, TorchModel


@MODELS.register_module('text-to-head', 'text_to_head')
class TextToHeadModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        """The HeadReconModel is implemented based on HRN, publicly available at
        https://github.com/youngLBW/HRN

        Args:
            model_dir: the root directory of the model files
        """
        super().__init__(model_dir, *args, **kwargs)

        self.model_dir = model_dir

        base_model_path = os.path.join(model_dir, 'base_model')
        controlnet_path = os.path.join(model_dir, 'control_net')

        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16)
        self.face_gen_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16)
        self.face_gen_pipeline.scheduler = DDIMScheduler.from_config(
            self.face_gen_pipeline.scheduler.config)
        self.face_gen_pipeline.enable_model_cpu_offload()

        self.add_prompt = ', 4K, good looking face, epic realistic, Sony a7, sharp, ' \
                          'skin detail pores, soft light, uniform illumination'
        self.neg_prompt = 'ugly, cross eye, bangs, teeth, glasses, hat, dark, shadow'

        control_pose_path = os.path.join(self.model_dir, 'control_pose.jpg')
        self.control_pose = load_image(control_pose_path)

    def forward(self, input):
        prompt = input['text'] + self.add_prompt
        image = self.face_gen_pipeline(
            prompt,
            negative_prompt=self.neg_prompt,
            image=self.control_pose,
            num_inference_steps=20).images[0]  # PIL.Image

        return image
