# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F

from modelscope.metainfo import Pipelines
from modelscope.models.cv.super_resolution import RRDBNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_super_resolution, module_name=Pipelines.image_super_resolution)
class ImageSuperResolutionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image super resolution pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.num_feat = 64
        self.num_block = 23
        self.scale = 4
        self.sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=self.num_feat,
            num_block=self.num_block,
            num_grow_ch=32,
            scale=self.scale).to(self.device)

        model_path = f'{self.model}/{ModelFile.TORCH_MODEL_FILE}'
        self.sr_model.load_state_dict(torch.load(model_path), strict=True)

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        img = torch.from_numpy(img).to(self.device).permute(
            2, 0, 1).unsqueeze(0) / 255.
        result = {'img': img}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        self.sr_model.eval()

        img = input['img']
        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')

        with torch.no_grad():
            output = self.sr_model(img)
            del img
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

        return {OutputKeys.OUTPUT_IMG: output}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
