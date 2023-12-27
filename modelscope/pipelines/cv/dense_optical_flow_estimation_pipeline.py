# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import flow_to_color, InputPadder
from modelscope.utils.logger import get_logger

logger = get_logger()

# class InputPadder:
#     """ Pads images such that dimensions are divisible by 8 """
#     def __init__(self, dims, mode='sintel'):
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
#         pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
#         if mode == 'sintel':
#             self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
#         else:
#             self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
# 
#     def pad(self, *inputs):
#         return [F.pad(x, self._pad, mode='replicate') for x in inputs]
# 
#     def unpad(self,x):
#         ht, wd = x.shape[-2:]
#         c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
#         return x[..., c[0]:c[1], c[2]:c[3]]

@PIPELINES.register_module(
    Tasks.dense_optical_flow_estimation,
    module_name=Pipelines.dense_optical_flow_estimation)
class DenseOpticalFlowEstimationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('dense optical flow estimation model, pipeline init')

    def load_image(self, img_name):
        img = LoadImage.convert_to_ndarray(img_name).astype(np.float32)
        # H, W = 480, 640
        # img = cv2.resize(img, [W, H])
        img = img.transpose(2, 0, 1) / 255.0

        return img

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img1 = self.load_image(input[0])
        img2 = self.load_image(input[1])
        
        image1 = torch.from_numpy(img1)[None].cuda().float()
        image2 = torch.from_numpy(img2)[None].cuda().float()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # img = LoadImage.convert_to_ndarray(input).astype(np.float32)
        # H, W = 480, 640
        # img = cv2.resize(img, [W, H])
        # img = img.transpose(2, 0, 1) / 255.0
        # imgs = img[None, ...]

        data = {'image1': image1, 'image2': image2}

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        flows = results[OutputKeys.FLOWS]
        # if isinstance(depths, torch.Tensor):
        #     depths = depths.detach().cpu().squeeze().numpy()
        
        flows_color = flow_to_color(flows)
        outputs = {
            OutputKeys.FLOWS: flows,
            OutputKeys.FLOWS_COLOR: flows_color
        }

        return outputs
