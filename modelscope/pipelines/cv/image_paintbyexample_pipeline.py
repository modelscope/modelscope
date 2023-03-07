# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Resize

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_paintbyexample import \
    StablediffusionPaintbyexample
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_paintbyexample, module_name=Pipelines.image_paintbyexample)
class ImagePaintbyexamplePipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        assert isinstance(model, str), 'model must be a single str'
        from paint_ldm.models.diffusion.plms import PLMSSampler
        super().__init__(model=model, auto_collate=False, **kwargs)
        self.sampler = PLMSSampler(self.model.model)
        self.start_code = None

    def get_tensor(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [
                torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))
            ]
        return torchvision.transforms.Compose(transform_list)

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
            ]
        return torchvision.transforms.Compose(transform_list)

    def preprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(input['img'], str):
            image_name, mask_name, ref_name = input['img'], input[
                'mask'], input['reference']
            img = load_image(image_name).resize((512, 512))
            ref = load_image(ref_name).resize((224, 224))
            mask = load_image(mask_name).resize((512, 512)).convert('L')
        elif isinstance(input['img'], PIL.Image.Image):
            img = input['img'].convert('RGB').resize((512, 512))
            ref = input['reference'].convert('RGB').resize((224, 224))
            mask = input['mask'].resize((512, 512)).convert('L')
        else:
            raise TypeError(
                'input should be either str or PIL.Image, and both inputs should have the same type'
            )
        img = self.get_tensor()(img)
        img = img.unsqueeze(0)
        ref = self.get_tensor_clip()(ref)
        ref = ref.unsqueeze(0)
        mask = np.array(mask)[None, None]
        mask = 1 - mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        inpaint_image = img * mask
        test_model_kwargs = {}
        test_model_kwargs['inpaint_mask'] = mask.to(self.device)
        test_model_kwargs['inpaint_image'] = inpaint_image.to(self.device)
        test_model_kwargs['ref_tensor'] = ref.to(self.device)

        return test_model_kwargs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = self.perform_inference(input)
        return {OutputKeys.OUTPUT_IMG: result}

    def perform_inference(self, test_model_kwargs):
        with torch.no_grad():
            with self.model.model.ema_scope():
                ref_tensor = test_model_kwargs['ref_tensor']
                uc = self.model.model.learnable_vector
                c = self.model.model.get_learned_conditioning(
                    ref_tensor.to(torch.float32))
                c = self.model.model.proj_out(c)
                z_inpaint = self.model.model.encode_first_stage(
                    test_model_kwargs['inpaint_image'])
                z_inpaint = self.model.model.get_first_stage_encoding(
                    z_inpaint).detach()
                test_model_kwargs['inpaint_image'] = z_inpaint
                test_model_kwargs['inpaint_mask'] = Resize(
                    [z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])

                shape = [4, 512 // 8, 512 // 8]
                samples_ddim, _ = self.sampler.sample(
                    S=50,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=5,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=self.start_code,
                    test_model_kwargs=test_model_kwargs)

                x_samples_ddim = self.model.model.decode_first_stage(
                    samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3,
                                                              1).numpy()

                x_checked_image = x_samples_ddim
                x_checked_image_torch = torch.from_numpy(
                    x_checked_image).permute(0, 3, 1, 2)[0]

                x_sample = 255. * rearrange(
                    x_checked_image_torch.cpu().numpy(), 'c h w -> h w c')
                img = x_sample.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
