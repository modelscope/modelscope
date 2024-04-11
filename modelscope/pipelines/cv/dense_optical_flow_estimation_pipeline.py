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
from modelscope.utils.cv.image_utils import InputPadder, flow_to_color
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.dense_optical_flow_estimation,
    module_name=Pipelines.dense_optical_flow_estimation)
class DenseOpticalFlowEstimationPipeline(Pipeline):
    r""" Card Detection Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> estimator = pipeline(Tasks.dense_optical_flow_estimation, model='Damo_XR_Lab/cv_raft_dense-optical-flow_things')
    >>> estimator([[
    >>>         'modelscope/models/cv/dense_optical_flow_estimation/data/test/images/dense_flow1.png',
    >>>         'modelscope/models/cv/dense_optical_flow_estimation/data/test/images/dense_flow2.png'
    >>>          ]])
    >>> [{'flows': tensor([[[[-1.6319, -1.6348, -1.6363,  ..., -1.7191, -1.7136, -1.7085],
    >>>           [-1.6324, -1.6344, -1.6351,  ..., -1.7110, -1.7048, -1.7005],
    >>>           [-1.6318, -1.6326, -1.6329,  ..., -1.7080, -1.7050, -1.7031],
    >>>           ...,
    >>>           [-2.0998, -2.1007, -2.0958,  ..., -1.4086, -1.4055, -1.3996],
    >>>           [-2.1043, -2.1031, -2.0988,  ..., -1.4075, -1.4049, -1.3991],
    >>>           [-2.1016, -2.0985, -2.0939,  ..., -1.4062, -1.4029, -1.3969]],
    >>>
    >>>          [[ 0.0343,  0.0386,  0.0401,  ...,  0.8053,  0.8050,  0.8057],
    >>>           [ 0.0311,  0.0354,  0.0369,  ...,  0.8004,  0.8007,  0.8050],
    >>>           [ 0.0274,  0.0309,  0.0322,  ...,  0.8007,  0.8016,  0.8080],
    >>>           ...,
    >>>           [ 0.5685,  0.5785,  0.5740,  ...,  0.4003,  0.4153,  0.4365],
    >>>           [ 0.5994,  0.6000,  0.5899,  ...,  0.4057,  0.4218,  0.4447],
    >>>           [ 0.6137,  0.6076,  0.5920,  ...,  0.4147,  0.4299,  0.4538]]]],
    >>>        device='cuda:0'), 'flows_color': array([[[255, 249, 219],
    >>>         [255, 249, 219],
    >>>         [255, 249, 219],
    >>>         ...,
    >>>         [236, 255, 213],
    >>>         [236, 255, 213],
    >>>         [236, 255, 213]],
    >>>
    >>>        [[255, 249, 219],
    >>>         [255, 249, 219],
    >>>         [255, 249, 219],
    >>>         ...,
    >>>         [236, 255, 213],
    >>>         [236, 255, 213],
    >>>         [236, 255, 213]],
    >>>
    >>>        [[255, 249, 219],
    >>>         [255, 249, 219],
    >>>         [255, 249, 219],
    >>>         ...,
    >>>         [236, 255, 213],
    >>>         [236, 255, 213],
    >>>         [236, 255, 213]],
    >>>
    >>>        ...,
    >>>
    >>>        [[251, 255, 207],
    >>>         [251, 255, 207],
    >>>         [251, 255, 207],
    >>>         ...,
    >>>         [251, 255, 222],
    >>>         [251, 255, 222],
    >>>         [250, 255, 222]],
    >>>
    >>>        [[250, 255, 207],
    >>>         [250, 255, 207],
    >>>         [250, 255, 207],
    >>>         ...,
    >>>         [251, 255, 222],
    >>>         [250, 255, 222],
    >>>         [249, 255, 222]],
    >>>
    >>>        [[249, 255, 207],
    >>>         [249, 255, 207],
    >>>         [250, 255, 207],
    >>>         ...,
    >>>         [251, 255, 222],
    >>>         [250, 255, 222],
    >>>         [249, 255, 222]]], dtype=uint8)}]
    """

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
        flow_ups = self.model.inference(input)
        results = flow_ups[-1]

        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out = self.model.postprocess(inputs)
        flows_color = flow_to_color([out[OutputKeys.FLOWS]])
        flows_color = flows_color[:, :, [2, 1, 0]]
        outputs = {
            OutputKeys.FLOWS: out[OutputKeys.FLOWS],
            OutputKeys.FLOWS_COLOR: flows_color
        }

        return outputs
