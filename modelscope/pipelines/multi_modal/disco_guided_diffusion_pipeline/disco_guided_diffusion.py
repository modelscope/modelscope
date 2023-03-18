# This code is borrowed and modified from Guided Diffusion Model,
# made publicly available under MIT license at
# https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/disco_project

import gc
import importlib
import math
import os

import clip
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal.guided_diffusion.script import \
    create_diffusion
from modelscope.models.multi_modal.guided_diffusion.unet import HFUNetModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks
from .utils import resize


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0,
                       torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


class MakeCutoutsDango(nn.Module):

    def __init__(
        self,
        cut_size,
        Overview=4,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
    ):
        super().__init__()
        self.padargs = {}
        self.cutout_debug = False
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(input,
                          ((sideY - max_size) // 2, (sideY - max_size) // 2,
                           (sideX - max_size) // 2, (sideX - max_size) // 2),
                          **self.padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if self.cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save(
                    'cutout_overview0.jpg', quality=99)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([])**self.IC_Size_Pow * (max_size - min_size)
                    + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size,
                               offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if self.cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save(
                    'cutout_InnerCrop.jpg', quality=99)
        cutouts = torch.cat(cutouts)

        cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711])


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.disco_guided_diffusion)
class DiscoDiffusionPipeline(DiffusersPipeline):

    def __init__(self, model: str, device: str = 'gpu', **kwargs):
        """  Chinese Disco Diffusion Pipeline.

        Examples:

        >>> import cv2
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.utils.constant import Tasks

        >>> prompt = '赛博朋克，城市'
        >>> output_image_path = './result.png'
        >>> input = {
        >>>     'text': prompt
        >>> }
        >>> pipe = pipeline(
        >>>     Tasks.text_to_image_synthesis,
        >>>     model='yyqoni/yinyueqin_cyberpunk',
        >>>     model_revision='v1.0')
        >>> output = pipe(input)['output_imgs'][0]
        >>> cv2.imwrite(output_image_path, output)
        >>> print('pipeline: the output image path is {}'.format(output_image_path))
        """

        super().__init__(model, device, **kwargs)

        model_path = model

        model_config = {'steps': 100, 'use_fp16': True}
        self.diffusion = create_diffusion(model_config)

        self.unet = HFUNetModel.from_pretrained(f'{model_path}/unet')

        self.unet.requires_grad_(False).eval().to(self.device)
        for name, param in self.unet.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if model_config['use_fp16']:
            self.unet.convert_to_fp16()

        with open(
                os.path.join(model_path, 'model_index.json'),
                'r',
                encoding='utf-8') as reader:
            text = reader.read()
        config_dict = json.loads(text)

        library = importlib.import_module(config_dict['tokenizer'][0])
        class_name = config_dict['tokenizer'][1]

        self.taiyi_tokenizer = getattr(
            library, class_name).from_pretrained(f'{model_path}/tokenizer')

        library = importlib.import_module(config_dict['text_encoder'][0])
        class_name = config_dict['text_encoder'][1]

        self.taiyi_transformer = getattr(library, class_name).from_pretrained(
            f'{model_path}/text_encoder').eval().to(self.device)

        self.clip_models = []
        self.clip_models.append(
            clip.load('ViT-L/14',
                      jit=False)[0].eval().requires_grad_(False).to(
                          self.device))

    def forward(self,
                inputs,
                init=None,
                init_scale=2000,
                skip_steps=10,
                randomize_class=True,
                eta=0.8,
                output_type='pil',
                return_dict=True,
                clip_guidance_scale=7500):
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')

        batch_size = 1
        cutn_batches = 1

        tv_scale = 0
        range_scale = 150
        sat_scale = 0

        cut_overview = eval('[12]*400+[4]*600')
        cut_innercut = eval('[4]*400+[12]*600')
        cut_ic_pow = eval('[1]*1000')
        cut_icgray_p = eval('[0.2]*400+[0]*600')

        side_x = 512
        side_y = 512

        if 'width' in inputs:
            side_x = inputs['width']
        if 'height' in inputs:
            side_y = inputs['height']
        frame_prompt = [inputs.get('text')]
        loss_values = []

        model_stats = []
        for clip_model in self.clip_models:
            # cutn = 16
            model_stat = {
                'clip_model': None,
                'target_embeds': [],
                'make_cutouts': None,
                'weights': []
            }
            model_stat['clip_model'] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                # NOTE use chinese CLIP
                txt = self.taiyi_transformer(
                    self.taiyi_tokenizer(txt,
                                         return_tensors='pt')['input_ids'].to(
                                             self.device)).logits

                model_stat['target_embeds'].append(txt)
                model_stat['weights'].append(weight)

            model_stat['target_embeds'] = torch.cat(
                model_stat['target_embeds'])
            model_stat['weights'] = torch.tensor(
                model_stat['weights'], device=self.device)
            if model_stat['weights'].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat['weights'] /= model_stat['weights'].sum().abs()
            model_stats.append(model_stat)

        init = None
        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]

                my_t = torch.ones([n], device=self.device,
                                  dtype=torch.long) * cur_t
                out = self.diffusion.p_mean_variance(
                    self.unet,
                    x,
                    my_t,
                    clip_denoised=False,
                    model_kwargs={'y': y})
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

                for model_stat in model_stats:
                    for i in range(cutn_batches):
                        t_int = int(t.item()) + 1
                        input_resolution = model_stat[
                            'clip_model'].visual.input_resolution

                        cuts = MakeCutoutsDango(
                            input_resolution,
                            Overview=cut_overview[1000 - t_int],
                            InnerCrop=cut_innercut[1000 - t_int],
                            IC_Size_Pow=cut_ic_pow[1000 - t_int],
                            IC_Grey_P=cut_icgray_p[1000 - t_int],
                        )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat['clip_model'].encode_image(
                            clip_in).float()
                        dists = spherical_dist_loss(
                            image_embeds.unsqueeze(1),
                            model_stat['target_embeds'].unsqueeze(0))
                        dists = dists.view([
                            cut_overview[1000 - t_int]
                            + cut_innercut[1000 - t_int], n, -1
                        ])
                        losses = dists.mul(
                            model_stat['weights']).sum(2).mean(0)
                        loss_values.append(losses.sum().item(
                        ))  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(
                            losses.sum() * clip_guidance_scale,
                            x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum(
                ) * range_scale + sat_losses.sum() * sat_scale
                if init is not None and init_scale:
                    init_losses = self.lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if not torch.isnan(x_in_grad).any():
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if not x_is_NaN:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=0.05) / magnitude
            return grad

        sample_fn = self.diffusion.ddim_sample_loop_progressive

        n_batches = 1

        for i in range(n_batches):
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = self.diffusion.num_timesteps - skip_steps - 1

            samples = sample_fn(
                self.unet,
                (batch_size, 3, side_y, side_x),
                clip_denoised=False,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=randomize_class,
                eta=eta,
            )

            for j, sample in enumerate(samples):
                image = sample['pred_xstart']
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()

            if output_type == 'pil':
                image = self.numpy_to_pil(image)
                return image

            if not return_dict:
                return (image, None)

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype('uint8')
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode='L') for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def postprocess(self, inputs):
        images = []
        for img in inputs:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}
