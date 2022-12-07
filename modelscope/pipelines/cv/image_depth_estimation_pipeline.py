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
from modelscope.utils.cv.image_utils import depth_to_color
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_depth_estimation, module_name=Pipelines.image_depth_estimation)
class ImageDepthEstimationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('depth estimation model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input).astype(np.float32)
        H, W = 480, 640
        img = cv2.resize(img, [W, H])
        img = img.transpose(2, 0, 1) / 255.0
        imgs = img[None, ...]
        data = {'imgs': imgs}

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        depths = results[OutputKeys.DEPTHS]
        if isinstance(depths, torch.Tensor):
            depths = depths.detach().cpu().squeeze().numpy()
        depths_color = depth_to_color(depths)
        outputs = {
            OutputKeys.DEPTHS: depths,
            OutputKeys.DEPTHS_COLOR: depths_color
        }

        return outputs
