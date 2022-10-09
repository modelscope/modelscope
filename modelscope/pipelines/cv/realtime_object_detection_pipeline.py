# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, List, Union

import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.realtime_object_detection import RealtimeDetector
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import load_image
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_object_detection,
    module_name=Pipelines.realtime_object_detection)
class RealtimeObjectDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.model = RealtimeDetector(model)

    def preprocess(self, input: Input) -> Dict[Tensor, Union[str, np.ndarray]]:
        output = self.model.preprocess(input)
        return {'pre_output': output}

    def forward(self, input: Tensor) -> Dict[Tensor, Dict[str, np.ndarray]]:
        pre_output = input['pre_output']
        forward_output = self.model(pre_output)
        return {'forward_output': forward_output}

    def postprocess(self, input: Dict[Tensor, Dict[str, np.ndarray]],
                    **kwargs) -> str:
        forward_output = input['forward_output']
        bboxes, scores, labels = forward_output
        return {
            OutputKeys.BOXES: bboxes,
            OutputKeys.SCORES: scores,
            OutputKeys.LABELS: labels,
        }
