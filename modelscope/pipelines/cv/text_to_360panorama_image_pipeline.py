# Copyright © Alibaba, Inc. and its affiliates.
import random
from typing import Any, Dict

import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import (ControlNetModel, DiffusionPipeline,
                       EulerAncestralDiscreteScheduler,
                       UniPCMultistepScheduler)
from PIL import Image
from realesrgan import RealESRGANer

from modelscope.metainfo import Pipelines
from modelscope.models.cv.text_to_360panorama_image import (
    StableDiffusionBlendExtendPipeline,
    StableDiffusionControlNetImg2ImgPanoPipeline)
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_to_360panorama_image,
    module_name=Pipelines.text_to_360panorama_image)
class Text2360PanoramaImagePipeline(Pipelines):
    """ Stable Diffusion for 360 Panorama Image Generation Pipeline.
    Example:
    >>> import cv2
    >>> from modelscope.outputs import OutputKeys
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> prompt = 'The mountains'
    >>> input = {'prompt': prompt, 'upscale': True}
    >>> model_id = 'damo/cv_diffusion_text-to-360panorama-image_generation'
    >>> txt2panoimg = pipeline(Tasks.text_to_360panorama_image, model=model_id, model_revision='v1.0.0')
    >>> output = txt2panoimg(input)[OutputKeys.OUTPUT_IMG]
    >>> cv2.imwrite('result.png', output)
    """

    def __init__(self, model: str, device: str = 'cuda', **kwargs):
        """
        Use `model` to create a stable diffusion pipeline for 360 panorama image generation.
        Args:
            model: model id on modelscope hub.
            device: str = 'cuda'
        """
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
                              ) if device is None else device
        if device == 'gpu':
            device = 'cuda'

        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        enable_xformers_memory_efficient_attention = kwargs.get(
            'enable_xformers_memory_efficient_attention', True)

        model_id = model + '/sd-base/'

        # init base model
        self.pipe = StableDiffusionBlendExtendPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype).to(device)
        self.pipe.vae.enable_tiling()
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config)
        # remove following line if xformers is not installed
        try:
            if enable_xformers_memory_efficient_attention:
                self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
        self.pipe.enable_model_cpu_offload()

        # init controlnet-sr model
        base_model_path = model + '/sr-base'
        controlnet_path = model + '/sr-control'
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch_dtype)
        self.pipe_sr = StableDiffusionControlNetImg2ImgPanoPipeline.from_pretrained(
            base_model_path, controlnet=controlnet,
            torch_dtype=torch_dtype).to(device)
        self.pipe_sr.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe_sr.vae.enable_tiling()
        # remove following line if xformers is not installed
        try:
            if enable_xformers_memory_efficient_attention:
                self.pipe_sr.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
        self.pipe_sr.enable_model_cpu_offload()

        # init realesrgan model
        sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2)
        netscale = 2

        model_path = model + '/RealESRGAN_x2plus.pth'

        dni_weight = None
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=sr_model,
            tile=384,
            tile_pad=20,
            pre_pad=20,
            half=False,
            device=device,
        )

    @staticmethod
    def blend_h(a, b, blend_extent):
        blend_extent = min(a.shape[1], b.shape[1], blend_extent)
        for x in range(blend_extent):
            b[:, x, :] = a[:, -blend_extent
                           + x, :] * (1 - x / blend_extent) + b[:, x, :] * (
                               x / blend_extent)
        return b

    def __call__(self, inputs: Dict[str, Any],
                 **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        num_inference_steps = inputs.get('num_inference_steps', 20)
        guidance_scale = inputs.get('guidance_scale', 7.5)
        preset_a_prompt = 'photorealistic, trend on artstation, ((best quality)), ((ultra high res))'
        add_prompt = inputs.get('add_prompt', preset_a_prompt)
        preset_n_prompt = 'persons, complex texture, small objects, sheltered, blur, worst quality, '\
                          'low quality, zombie, logo, text, watermark, username, monochrome, '\
                          'complex lighting'
        negative_prompt = inputs.get('negative_prompt', preset_n_prompt)
        seed = inputs.get('seed', -1)
        upscale = inputs.get('upscale', True)
        refinement = inputs.get('refinement', True)

        if 'prompt' in inputs.keys():
            prompt = inputs['prompt']
        else:
            # for demo_service
            prompt = forward_params.get('prompt', 'the living room')

        print(f'Test with prompt: {prompt}')

        if seed == -1:
            seed = random.randint(0, 65535)
        print(f'global seed: {seed}')

        generator = torch.manual_seed(seed)

        prompt = '<360panorama>, ' + prompt + ', ' + add_prompt
        output_img = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=512,
            width=1024,
            guidance_scale=guidance_scale,
            generator=generator).images[0]

        if not upscale:
            print('finished')
        else:
            print('inputs: upscale=True, running upscaler.')
            print('running upscaler step1. Initial super-resolution')
            sr_scale = 2.0
            output_img = self.pipe_sr(
                prompt.replace('<360panorama>, ', ''),
                negative_prompt=negative_prompt,
                image=output_img.resize(
                    (int(1536 * sr_scale), int(768 * sr_scale))),
                num_inference_steps=7,
                generator=generator,
                control_image=output_img.resize(
                    (int(1536 * sr_scale), int(768 * sr_scale))),
                strength=0.8,
                controlnet_conditioning_scale=1.0,
                guidance_scale=15,
            ).images[0]

            print('running upscaler step2. Super-resolution with Real-ESRGAN')
            output_img = output_img.resize((1536 * 2, 768 * 2))
            w = output_img.size[0]
            blend_extend = 10
            outscale = 2
            output_img = np.array(output_img)
            output_img = np.concatenate(
                [output_img, output_img[:, :blend_extend, :]], axis=1)
            output_img, _ = self.upsampler.enhance(
                output_img, outscale=outscale)
            output_img = self.blend_h(output_img, output_img,
                                      blend_extend * outscale)
            output_img = Image.fromarray(output_img[:, :w * outscale, :])

            if refinement:
                print(
                    'inputs: refinement=True, running refinement. This is a bit time-consuming.'
                )
                sr_scale = 4
                output_img = self.pipe_sr(
                    prompt.replace('<360panorama>, ', ''),
                    negative_prompt=negative_prompt,
                    image=output_img.resize(
                        (int(1536 * sr_scale), int(768 * sr_scale))),
                    num_inference_steps=7,
                    generator=generator,
                    control_image=output_img.resize(
                        (int(1536 * sr_scale), int(768 * sr_scale))),
                    strength=0.8,
                    controlnet_conditioning_scale=1.0,
                    guidance_scale=17,
                ).images[0]
            print('finished')

        output_img = np.array(output_img)
        return {'output_img': output_img[:, :, ::-1]}
