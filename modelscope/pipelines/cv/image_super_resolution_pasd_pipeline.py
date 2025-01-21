# Copyright © Alibaba, Inc. and its affiliates.

import os
import tempfile
from typing import Any, Dict, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_portrait_enhancement.retinaface import \
    detection
from modelscope.models.cv.image_super_resolution_pasd.misc import (
    load_dreambooth_lora, wavelet_color_fix)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.pasd_pipeline import \
    PixelAwareStableDiffusionPipeline
from modelscope.preprocessors.image import load_image
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device


@PIPELINES.register_module(
    Tasks.image_super_resolution_pasd,
    module_name=Pipelines.image_super_resolution_pasd)
class ImageSuperResolutionPASDPipeline(Pipeline):
    """ Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution Pipeline.

    Example:

    >>> import cv2
    >>> from modelscope.outputs import OutputKeys
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> input_location = 'example_image.png'

    >>> input = {
    >>>     'image': input_location,
    >>>     'upscale': 2,
    >>>     'prompt': prompt,
    >>>     'fidelity_scale_fg': 1.0,
    >>>     'fidelity_scale_bg': 1.0
    >>> }
    >>> pasd = pipeline(Tasks.image_super_resolution_pasd, model='damo/PASD_image_super_resolutions')
    >>> output = pasd(input)[OutputKeys.OUTPUT_IMG]
    >>> cv2.imwrite('result.png', output)

    """

    def __init__(self, model: str, device_name: str = 'cuda', **kwargs):
        """
        use `model` to create a image super resolution pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        self.device = create_device(device_name)
        self.config = Config.from_file(
            os.path.join(model, ModelFile.CONFIGURATION))
        version = self.config.pipeline.get('version', 'pasd_v2')
        if version == 'pasd':
            from modelscope.models.cv.image_super_resolution_pasd import (
                ControlNetModel, UNet2DConditionModel)
        else:
            from modelscope.models.cv.image_super_resolution_pasd_v2 import (
                ControlNetModel, UNet2DConditionModel)
        cfg = self.config.model_cfg
        dreambooth_lora_ckpt = cfg['dreambooth_lora_ckpt']
        tiled_size = cfg['tiled_size']
        self.process_size = cfg['process_size']

        scheduler = UniPCMultistepScheduler.from_pretrained(
            model, subfolder='scheduler')
        text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder='text_encoder')
        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
        vae = AutoencoderKL.from_pretrained(model, subfolder='vae')
        feature_extractor = CLIPImageProcessor.from_pretrained(
            f'{model}/feature_extractor')
        unet = UNet2DConditionModel.from_pretrained(model, subfolder='unet')
        controlnet = ControlNetModel.from_pretrained(
            model, subfolder='controlnet')

        unet, vae = load_dreambooth_lora(unet, vae,
                                         f'{model}/{dreambooth_lora_ckpt}')

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        controlnet.requires_grad_(False)

        text_encoder.to(self.device, dtype=torch_dtype)
        vae.to(self.device, dtype=torch_dtype)
        unet.to(self.device, dtype=torch_dtype)
        controlnet.to(self.device, dtype=torch_dtype)

        self.pipeline = PixelAwareStableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipeline._init_tiled_vae(decoder_tile_size=tiled_size)
        self.pipeline.enable_model_cpu_offload()

        self.weights = ResNet50_Weights.DEFAULT
        self.resnet_preprocess = self.weights.transforms()
        self.resnet = resnet50(weights=self.weights)
        self.resnet.eval()

        self.threshold = 0.8
        detector_model_path = f'{model}/RetinaFace-R50.pth'
        self.face_detector = detection.RetinaFaceDetection(
            detector_model_path, self.device)

        self.resize_preproc = transforms.Compose([
            transforms.Resize(
                self.process_size,
                interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    def preprocess(self, input: Input):
        return input

    def forward(self, inputs: Dict[str, Any]):
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )

        num_inference_steps = inputs.get('num_inference_steps', 20)
        guidance_scale = inputs.get('guidance_scale', 7.5)
        added_prompt = inputs.get(
            'added_prompt',
            'clean, high-resolution, 8k, best quality, masterpiece, extremely detailed'
        )
        negative_prompt = inputs.get(
            'negative_prompt',
            'dotted, noise, blur, lowres, smooth, longbody, bad anatomy, bad hands, missing fingers, extra digit, \
            fewer digits, cropped, worst quality, low quality')
        eta = inputs.get('eta', 0.0)
        prompt = inputs.get('prompt', '')
        upscale = inputs.get('upscale', 2)
        fidelity_scale_fg = inputs.get('fidelity_scale_fg', 1.0)
        fidelity_scale_bg = inputs.get('fidelity_scale_bg', 1.0)

        input_image = load_image(inputs['image']).convert('RGB')

        with torch.no_grad():
            generator = torch.Generator(device=self.device)

            batch = self.resnet_preprocess(input_image).unsqueeze(0)
            prediction = self.resnet(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = self.weights.meta['categories'][class_id]
            if score >= 0.1:
                prompt += f'{category_name}' if prompt == '' else f', {category_name}'

            prompt = added_prompt if prompt == '' else f'{prompt}, {added_prompt}'

            ori_width, ori_height = input_image.size
            resize_flag = True
            rscale = upscale

            input_image = input_image.resize(
                (input_image.size[0] * rscale, input_image.size[1] * rscale))

            if min(input_image.size) < self.process_size:
                input_image = self.resize_preproc(input_image)

            input_image = input_image.resize(
                (input_image.size[0] // 8 * 8, input_image.size[1] // 8 * 8))
            width, height = input_image.size

            fg_mask = None
            if fidelity_scale_fg != fidelity_scale_bg:
                fg_mask = torch.zeros([1, 1, height, width])
                facebs, _ = self.face_detector.detect(np.array(input_image))
                for fb in facebs:
                    if fb[-1] < self.threshold:
                        continue
                    fb = list(map(int, fb))
                    fg_mask[:, :, fb[1]:fb[3], fb[0]:fb[2]] = 1
                fg_mask = fg_mask.to(self.device)

            if fg_mask is None:
                fidelity_scale = min(
                    max(fidelity_scale_fg, fidelity_scale_bg), 1)
                fidelity_scale_fg = fidelity_scale_bg = fidelity_scale

            try:
                image = self.pipeline(
                    prompt,
                    input_image,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    conditioning_scale_fg=fidelity_scale_fg,
                    conditioning_scale_bg=fidelity_scale_bg,
                    fg_mask=fg_mask,
                    eta=eta,
                ).images[0]

                image = wavelet_color_fix(image, input_image)

                if resize_flag:
                    image = image.resize(
                        (ori_width * rscale, ori_height * rscale))
            except Exception as e:
                print(e)
                image = PIL.Image.new('RGB', (512, 512), (0, 0, 0))

        return {'result': image}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = np.array(inputs['result'])
        return {OutputKeys.OUTPUT_IMG: result[:, :, ::-1]}
