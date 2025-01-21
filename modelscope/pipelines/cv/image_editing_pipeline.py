# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_editing import (
    MutualSelfAttentionControl, register_attention_editor_diffusers)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ImageEditingPipeline']


@PIPELINES.register_module(
    Tasks.image_editing, module_name=Pipelines.image_editing)
class ImageEditingPipeline(DiffusersPipeline):

    def __init__(self, model=str, preprocessor=None, **kwargs):
        """  MasaCtrl Image Editing Pipeline.

        Examples:

        >>> import cv2
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.utils.constant import Tasks

        >>> prompts = [
        >>>     "",                           # source prompt
        >>>     "a photo of a running corgi"  # target prompt
        >>> ]
        >>> output_image_path = './result.png'
        >>> img = 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/public/ModelScope/test/images/corgi.jpg'
        >>> input = {'img': img, 'prompts': prompts}
        >>>
        >>> pipe = pipeline(
        >>>     Tasks.image_editing,
        >>>     model='damo/cv_masactrl_image-editing')
        >>>
        >>> output = pipe(input)['output_img']
        >>> cv2.imwrite(output_image_path, output)
        >>> print('pipeline: the output image path is {}'.format(output_image_path))
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        torch_dtype = kwargs.get('torch_dtype', torch.float32)
        self._device = getattr(
            kwargs, 'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info('load image editing pipeline done')
        scheduler = DDIMScheduler.from_pretrained(
            os.path.join(model, 'stable-diffusion-v1-4'),
            subfolder='scheduler')
        self.pipeline = _MasaCtrlPipeline.from_pretrained(
            os.path.join(model, 'stable-diffusion-v1-4'),
            scheduler=scheduler,
            torch_dtype=torch_dtype,
            use_safetensors=True).to(self._device)

    def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input.get('img'))
        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])  # [-1, 1]
        img = test_transforms(img).unsqueeze(0)
        img = F.interpolate(img, (512, 512))
        input['img'] = img.to(self._device)
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(input, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        prompts = input.get('prompts')
        start_code, latents_list = self.pipeline.invert(
            input.get('img'),
            prompts[0],
            guidance_scale=7.5,
            num_inference_steps=50,
            return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        STEP, LAYER = 4, 10
        editor = MutualSelfAttentionControl(STEP, LAYER)
        register_attention_editor_diffusers(self.pipeline, editor)

        # inference the synthesized image
        output = self.pipeline(
            prompts,
            latents=start_code,
            guidance_scale=input.get('guidance_scale', 7.5),
        )[-1:]

        return {'output_tensor': output}

    def postprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        output_img = (input['output_tensor'].squeeze(0) * 255).cpu().permute(
            1, 2, 0).numpy().astype('uint8')
        return {OutputKeys.OUTPUT_IMG: output_img[:, :, ::-1]}


class _MasaCtrlPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0,
        verbose=False,
    ):
        """
        Inverse sampling for DDIM Inversion
        x_t -> x_(t+1)
        """
        if verbose:
            print('timestep: ', timestep)
        next_step = timestep
        timestep = min(
            timestep - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float = 0.0,
        verbose=False,
    ):
        """
        predict the sample the next step in the denoise process.
        x_t -> x_(t-1)
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = self._execution_device
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='pt'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == 'pt':
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def __call__(self,
                 prompt,
                 batch_size=1,
                 height=512,
                 width=512,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 eta=0.0,
                 latents=None,
                 unconditioning=None,
                 neg_prompt=None,
                 ref_intermediate_latents=None,
                 return_intermediates=False,
                 **kwds):
        DEVICE = self._execution_device
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=77, return_tensors='pt')

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print('input text embeddings :', text_embeddings.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height // 8,
                         width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f'The shape of input latent tensor {latents.shape} should equal ' \
                                                   f'to predefined one.'

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ''
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding='max_length',
                max_length=77,
                return_tensors='pt')
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0)

        print('latents shape: ', latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(
                tqdm(self.scheduler.timesteps, desc='DDIM Sampler')):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([
                    unconditioning[i].expand(*text_embeddings.shape),
                    text_embeddings
                ])
            # predict the noise
            noise_pred = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type='pt')
        if return_intermediates:
            pred_x0_list = [
                self.latent2image(img, return_type='pt')
                for img in pred_x0_list
            ]
            latents_list = [
                self.latent2image(img, return_type='pt')
                for img in latents_list
            ]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(self,
               image: torch.Tensor,
               prompt,
               num_inference_steps=50,
               guidance_scale=7.5,
               eta=0.0,
               return_intermediates=False,
               **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = self._execution_device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=77, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print('input text embeddings :', text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            unconditional_input = self.tokenizer(
                [''] * batch_size,
                padding='max_length',
                max_length=77,
                return_tensors='pt')
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0)

        print('latents shape: ', latents.shape)
        self.scheduler.set_timesteps(num_inference_steps)
        print('Valid timesteps: ', reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(
                tqdm(
                    reversed(self.scheduler.timesteps),
                    desc='DDIM Inversion')):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            return latents, latents_list
        return latents, start_latents
