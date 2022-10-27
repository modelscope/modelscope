# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_human_hand_detection import det_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_human_hand_detection,
    module_name=Pipelines.face_human_hand_detection)
class NanoDettForFaceHumanHandDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create face-human-hand detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input['input_path'])
        return img

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        cls_list, bbox_list, score_list = det_infer.inference(
            self.model, self.device, input)
        logger.info(cls_list, bbox_list, score_list)
        return {
            OutputKeys.LABELS: cls_list,
            OutputKeys.BOXES: bbox_list,
            OutputKeys.SCORES: score_list
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
