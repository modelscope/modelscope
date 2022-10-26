# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_inpainting import FFTInpainting
from modelscope.models.cv.image_inpainting.refinement import refine_predict
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_inpainting, module_name=Pipelines.image_inpainting)
class ImageInpaintingPipeline(Pipeline):

    def __init__(self,
                 model: str,
                 pad_out_to_modulo=8,
                 refine=False,
                 **kwargs):
        """
            model: model id on modelscope hub.
        """
        assert isinstance(model, str), 'model must be a single str'
        super().__init__(model=model, auto_collate=False, **kwargs)
        self.refine = refine
        logger.info(f'loading model from dir {model}')
        self.infer_model = FFTInpainting(model, predict_only=True)
        if not self.refine:
            self.infer_model.to(self.device)
        self.infer_model.eval()
        logger.info(f'loading model done, refinement is set to {self.refine}')
        self.pad_out_to_modulo = pad_out_to_modulo

    def move_to_device(self, obj, device):
        if isinstance(obj, nn.Module):
            return obj.to(device)
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):
            return [self.move_to_device(el, device) for el in obj]
        if isinstance(obj, dict):
            return {
                name: self.move_to_device(val, device)
                for name, val in obj.items()
            }
        raise ValueError(f'Unexpected type {type(obj)}')

    def transforms(self, img):
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
        return out_img

    def ceil_modulo(self, x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def pad_img_to_modulo(self, img, mod):
        channels, height, width = img.shape
        out_height = self.ceil_modulo(height, mod)
        out_width = self.ceil_modulo(width, mod)
        return np.pad(
            img, ((0, 0), (0, out_height - height), (0, out_width - width)),
            mode='symmetric')

    def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(input['img'], str):
            image_name, mask_name = input['img'], input['mask']
            img = LoadImage.convert_to_ndarray(image_name)
            img = self.transforms(img)
            mask = np.array(LoadImage(mode='L')(mask_name)['img'])
            mask = self.transforms(mask)
        elif isinstance(input['img'], PIL.Image.Image):
            img = input['img']
            img = self.transforms(np.array(img))
            mask = input['mask'].convert('L')
            mask = self.transforms(np.array(mask))
        else:
            raise TypeError(
                'input should be either str or PIL.Image, and both inputs should have the same type'
            )
        result = dict(image=img, mask=mask[None, ...])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = self.pad_img_to_modulo(result['image'],
                                                     self.pad_out_to_modulo)
            result['mask'] = self.pad_img_to_modulo(result['mask'],
                                                    self.pad_out_to_modulo)

        # Since Pipeline use default torch.no_grad() for performing forward func.
        # We conduct inference here in case of doing training for refinement.
        result = self.perform_inference(result)
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return {OutputKeys.OUTPUT_IMG: input}

    def perform_inference(self, data):
        batch = default_collate([data])
        if self.refine:
            assert 'unpad_to_size' in batch, 'Unpadded size is required for the refinement'
            assert 'cuda' in str(self.device), 'GPU is required for refinement'
            gpu_ids = str(self.device).split(':')[-1]
            cur_res = refine_predict(
                batch,
                self.infer_model,
                gpu_ids=gpu_ids,
                modulo=self.pad_out_to_modulo,
                n_iters=15,
                lr=0.002,
                min_side=512,
                max_scales=3,
                px_budget=900000)
            cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                batch = self.move_to_device(batch, self.device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = self.infer_model(batch)
                cur_res = batch['inpainted'][0].permute(
                    1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
