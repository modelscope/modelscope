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
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_normal_estimation,
    module_name=Pipelines.image_normal_estimation)
class ImageNormalEstimationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image normal estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('normal estimation model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input).astype(np.float32)
        H, W = 384, 384
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
        normals = results[OutputKeys.NORMALS]
        if isinstance(normals, torch.Tensor):
            normals = normals.detach().cpu().squeeze().numpy()
        normals_color = (np.transpose(normals,
                                      (1, 2, 0)) * 255).astype(np.uint8)
        outputs = {
            OutputKeys.NORMALS: normals,
            OutputKeys.NORMALS_COLOR: normals_color
        }

        return outputs
