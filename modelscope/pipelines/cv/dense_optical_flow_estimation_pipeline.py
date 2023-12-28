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
        img = img.transpose(2, 0, 1)

        return img

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img1 = self.load_image(input[0])
        img2 = self.load_image(input[1])
        
        image1 = torch.from_numpy(img1)[None].cuda().float()
        image2 = torch.from_numpy(img2)[None].cuda().float()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        data = {'image1': image1, 'image2': image2}

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        flows = results[OutputKeys.FLOWS]
        
        flows_color = flow_to_color(flows)
        flows_color = flows_color[:,:,[2,1,0]]
        outputs = {
            OutputKeys.FLOWS: flows,
            OutputKeys.FLOWS_COLOR: flows_color
        }

        return outputs
