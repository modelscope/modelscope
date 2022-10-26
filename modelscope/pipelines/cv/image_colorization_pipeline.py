# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
from torchvision import models, transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_colorization import (DynamicUnetDeep,
                                                     DynamicUnetWide, NormType)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_colorization, module_name=Pipelines.image_colorization)
class ImageColorizationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image colorization pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.cut = 8
        self.size = 512
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.orig_img = None
        self.model_type = 'stable'
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

        if self.model_type == 'stable':
            body = models.resnet101(pretrained=True)
            body = torch.nn.Sequential(*list(body.children())[:self.cut])
            self.model = DynamicUnetWide(
                body,
                n_classes=3,
                blur=True,
                blur_final=True,
                self_attention=True,
                y_range=(-3.0, 3.0),
                norm_type=NormType.Spectral,
                last_cross=True,
                bottle=False,
                nf_factor=2,
            ).to(self.device)
        else:
            body = models.resnet34(pretrained=True)
            body = torch.nn.Sequential(*list(body.children())[:cut])
            self.model = DynamicUnetDeep(
                body,
                n_classes=3,
                blur=True,
                blur_final=True,
                self_attention=True,
                y_range=(-3.0, 3.0),
                norm_type=NormType.Spectral,
                last_cross=True,
                bottle=False,
                nf_factor=1.5,
            ).to(self.device)

        model_path = f'{model}/{ModelFile.TORCH_MODEL_FILE}'
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['model'],
            strict=True)

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input).convert('LA').convert('RGB')

        self.wide, self.height = img.size
        if self.wide * self.height < 100000:
            self.size = 256
        self.orig_img = img.copy()
        img = img.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        img = self.norm(img).unsqueeze(0).to(self.device)
        result = {'img': img}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(input['img'])[0]

        out = self.denorm(out)
        out = out.float().clamp(min=0, max=1)
        out_img = (out.permute(1, 2, 0).flip(2).cpu().numpy() * 255).astype(
            np.uint8)

        if self.orig_img is not None:
            color_np = cv2.resize(out_img, self.orig_img.size)
            orig_np = np.asarray(self.orig_img)
            color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
            orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
            hires = np.copy(orig_yuv)
            hires[:, :, 1:3] = color_yuv[:, :, 1:3]
            out_img = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)

        return {OutputKeys.OUTPUT_IMG: out_img.astype(np.uint8)}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
