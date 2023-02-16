# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.models.cv.facial_landmark_confidence import \
    FacialLandmarkConfidence
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
    Tasks.face_2d_keypoints, module_name=Pipelines.facial_landmark_confidence)
class FacialLandmarkConfidencePipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a facial landmrk confidence pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {ckpt_path}')
        flcm = FacialLandmarkConfidence(
            model_path=ckpt_path, device=self.device)
        self.flcm = flcm
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:

        result = super().preprocess(input)
        img = LoadImage.convert_to_ndarray(input)
        img = img[:, :, ::-1]
        result['orig_img'] = img.astype(np.float32)
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = self.flcm(input)
        assert result is not None
        lms = result[0].reshape(-1, 10).tolist()
        scores = [1 - result[1].tolist()]
        boxes = input['bbox'].cpu().numpy()[np.newaxis, :].tolist()
        output_poses = []
        return {
            OutputKeys.SCORES: scores,
            OutputKeys.POSES: output_poses,
            OutputKeys.KEYPOINTS: lms,
            OutputKeys.BOXES: boxes
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
