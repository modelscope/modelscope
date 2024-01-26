# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

import cv2
import einops
import numpy as np
import requests
import torch
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.models.cv.anydoor.cldm.ddim_hacked import DDIMSampler
from modelscope.models.cv.anydoor.datasets.data_utils import (
    box2squre, box_in_box, expand_bbox, expand_image_mask, get_bbox_from_mask,
    pad_to_square, sobel)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_to_image_generation, module_name=Pipelines.anydoor)
class AnydoorPipeline(Pipeline):
    r""" AnyDoor Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> from PIL import Image

    >>> ref_image = 'data/test/images/image_anydoor_fg.png'
    >>> ref_mask = 'data/test/images/image_anydoor_fg_mask.png'
    >>> bg_image = 'data/test/images/image_anydoor_bg.png'
    >>> bg_mask = 'data/test/images/image_anydoor_bg_mask.png'

    >>> anydoor_pipeline = pipeline(Tasks.image_to_image_generation, model='damo/AnyDoor')
    >>> out = anydoor_pipeline((ref_image, ref_mask, bg_image, bg_mask))
    >>> assert isinstance(out['output_img'], Image.Image)
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a action detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_ckpt = os.path.join(self.model.model_dir,
                                  self.cfg.model.model_path)
        self.model.load_state_dict(
            self._get_state_dict(model_ckpt, location='cuda'))
        self.ddim_sampler = DDIMSampler(self.model)

    @staticmethod
    def _get_state_dict(ckpt_path, location='cpu'):

        def get_state_dict(d):
            return d.get('state_dict', d)

        _, extension = os.path.splitext(ckpt_path)
        if extension.lower() == '.safetensors':
            import safetensors.torch
            state_dict = safetensors.torch.load_file(
                ckpt_path, device=location)
        else:
            state_dict = get_state_dict(
                torch.load(ckpt_path, map_location=torch.device(location)))
        state_dict = get_state_dict(state_dict)
        print(f'Loaded state_dict from [{ckpt_path}]')
        return state_dict

    def preprocess(self, inputs: Input) -> Dict[str, Any]:
        ref_image, ref_mask, tar_image, tar_mask = inputs
        ref_image = np.asarray(load_image(ref_image).convert('RGB'))
        ref_mask = np.where(
            np.asarray(load_image(ref_mask).convert('L')) > 128, 1,
            0).astype(np.uint8)
        tar_image = np.asarray(load_image(tar_image).convert('RGB'))
        tar_mask = np.where(
            np.asarray(load_image(tar_mask).convert('L')) > 128, 1,
            0).astype(np.uint8)

        # ========= Reference ===========
        # ref expand
        ref_box_yyxx = get_bbox_from_mask(ref_mask)

        # ref filter mask
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(
            ref_image) * 255 * (1 - ref_mask_3)

        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = ref_mask[y1:y2, x1:x2]

        ratio = np.random.randint(11, 15) / 10  # 11,13
        masked_ref_image, ref_mask = expand_image_mask(
            masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

        # to square and resize
        masked_ref_image = pad_to_square(
            masked_ref_image, pad_value=255, random=False)
        masked_ref_image = cv2.resize(
            masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8),
                                (224, 224)).astype(np.uint8)
        ref_mask = ref_mask_3[:, :, 0]

        # collage aug
        masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
        ref_mask_3 = np.stack(
            [ref_mask_compose, ref_mask_compose, ref_mask_compose], -1)
        ref_image_collage = sobel(masked_ref_image_compose,
                                  ref_mask_compose / 255)

        # ========= Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(
            tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])  # 1.1  1.3

        # crop
        tar_box_yyxx_crop = expand_bbox(
            tar_image, tar_box_yyxx, ratio=[1.3, 3.0])
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)  # crop box
        y1, y2, x1, x2 = tar_box_yyxx_crop

        cropped_target_image = tar_image[y1:y2, x1:x2, :]
        cropped_tar_mask = tar_mask[y1:y2, x1:x2]

        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx

        # collage
        ref_image_collage = cv2.resize(
            ref_image_collage.astype(np.uint8), (x2 - x1, y2 - y1))
        ref_mask_compose = cv2.resize(
            ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy()
        collage[y1:y2, x1:x2, :] = ref_image_collage

        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2, x1:x2, :] = 1.0
        collage_mask = np.stack(
            [cropped_tar_mask, cropped_tar_mask, cropped_tar_mask], -1)

        # the size before pad
        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(
            cropped_target_image, pad_value=0, random=False).astype(np.uint8)
        collage = pad_to_square(
            collage, pad_value=0, random=False).astype(np.uint8)
        collage_mask = pad_to_square(
            collage_mask, pad_value=0, random=False).astype(np.uint8)

        # the size after pad
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(
            cropped_target_image.astype(np.uint8),
            (512, 512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8),
                             (512, 512)).astype(np.float32)
        collage_mask = (cv2.resize(collage_mask.astype(
            np.uint8), (512, 512)).astype(np.float32) > 0.5).astype(np.float32)

        masked_ref_image = masked_ref_image / 255
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0
        collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

        item = dict(
            tar_image=tar_image,
            ref=masked_ref_image.copy(),
            jpg=cropped_target_image.copy(),
            hint=collage.copy(),
            extra_sizes=np.array([H1, W1, H2, W2]),
            tar_box_yyxx_crop=np.array(tar_box_yyxx_crop))
        return item

    def forward(self,
                item: Dict[str, Any],
                num_samples=1,
                strength=1.0,
                ddim_steps=30,
                scale=3.0) -> Dict[str, Any]:
        tar_image = item['tar_image'].cpu().numpy()
        ref = item['ref']
        hint = item['hint']
        num_samples = 1

        control = hint.float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        clip_input = ref.float().cuda()
        clip_input = torch.stack([clip_input for _ in range(num_samples)],
                                 dim=0)
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

        H, W = 512, 512

        cond = {
            'c_concat': [control],
            'c_crossattn': [self.model.get_learned_conditioning(clip_input)]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [torch.zeros((1, 3, 224, 224))] * num_samples)
            ]
        }
        shape = (4, H // 8, W // 8)

        self.model.control_scales = ([strength] * 13)
        samples, _ = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5
                     + 127.5).cpu().numpy()

        result = x_samples[0][:, :, ::-1]
        result = np.clip(result, 0, 255)

        pred = x_samples[0]
        pred = np.clip(pred, 0, 255)[1:, :, :]
        sizes = item['extra_sizes'].cpu().numpy()
        tar_box_yyxx_crop = item['tar_box_yyxx_crop'].cpu().numpy()
        return dict(
            pred=pred,
            tar_image=tar_image,
            sizes=sizes,
            tar_box_yyxx_crop=tar_box_yyxx_crop)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pred = inputs['pred']
        tar_image = inputs['tar_image']
        extra_sizes = inputs['sizes']
        tar_box_yyxx_crop = inputs['tar_box_yyxx_crop']

        H1, W1, H2, W2 = extra_sizes
        y1, y2, x1, x2 = tar_box_yyxx_crop
        pred = cv2.resize(pred, (W2, H2))
        m = 3  # maigin_pixel

        if W1 == H1:
            tar_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
            gen_image = torch.from_numpy(tar_image.copy()).permute(2, 0, 1)
            gen_image = gen_image.permute(1, 2, 0).numpy()
            gen_image = Image.fromarray(gen_image, mode='RGB')
            return {OutputKeys.OUTPUT_IMG: gen_image}

        if W1 < W2:
            pad1 = int((W2 - W1) / 2)
            pad2 = W2 - W1 - pad1
            pred = pred[:, pad1:-pad2, :]
        else:
            pad1 = int((H2 - H1) / 2)
            pad2 = H2 - H1 - pad1
            pred = pred[pad1:-pad2, :, :]

        gen_image = tar_image.copy()
        gen_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]

        gen_image = torch.from_numpy(gen_image).permute(2, 0, 1)
        gen_image = gen_image.permute(1, 2, 0).numpy()
        gen_image = Image.fromarray(gen_image, mode='RGB')
        return {OutputKeys.OUTPUT_IMG: gen_image}
