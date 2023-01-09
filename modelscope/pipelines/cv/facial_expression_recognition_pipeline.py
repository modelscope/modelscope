# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.models.cv.facial_expression_recognition import \
    FacialExpressionRecognition
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from . import FaceProcessingBasePipeline

logger = get_logger()


@PIPELINES.register_module(
    Tasks.facial_expression_recognition,
    module_name=Pipelines.facial_expression_recognition)
class FacialExpressionRecognitionPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {ckpt_path}')
        device = torch.device(
            f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        fer = FacialExpressionRecognition(model_path=ckpt_path, device=device)
        self.fer = fer
        self.device = device
        logger.info('load model done')

        self.map_list = [
            'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ]

    def preprocess(self, input: Input) -> Dict[str, Any]:
        result = super(FacialExpressionRecognitionPipeline,
                       self).preprocess(input)
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = self.fer(input)
        assert result is not None
        scores = result[0].tolist()
        return {OutputKeys.SCORES: scores, OutputKeys.LABELS: self.map_list}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
