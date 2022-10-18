# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import cv2
import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import \
    show_image_object_detection_auto_result
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.tinynas_detection)
class TinynasDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, input: Input) -> Dict[str, Any]:

        img = LoadImage.convert_to_ndarray(input)
        self.img = img
        img = img.astype(np.float)
        img = self.model.preprocess(img)
        result = {'img': img.to(self.device)}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        outputs = self.model.inference(input['img'])
        result = {'data': outputs}
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        bboxes, scores, labels = self.model.postprocess(inputs['data'])
        if bboxes is None:
            outputs = {
                OutputKeys.SCORES: [],
                OutputKeys.LABELS: [],
                OutputKeys.BOXES: []
            }
        else:
            outputs = {
                OutputKeys.SCORES: scores,
                OutputKeys.LABELS: labels,
                OutputKeys.BOXES: bboxes
            }
        return outputs

    def show_result(self, img_path, result, save_path=None):
        show_image_object_detection_auto_result(img_path, result, save_path)
