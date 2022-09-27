# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_detection import MtcnnFaceDetector
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_detection, module_name=Pipelines.mtcnn_face_detection)
class MtcnnFaceDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face detection pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, './weights')
        logger.info(f'loading model from {ckpt_path}')
        device = torch.device(
            f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        detector = MtcnnFaceDetector(model_path=ckpt_path, device=device)
        self.detector = detector
        self.device = device
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = self.detector(input)
        assert result is not None
        bboxes = result[0][:, :4].tolist()
        scores = result[0][:, 4].tolist()
        lms = result[1].tolist()
        return {
            OutputKeys.SCORES: scores,
            OutputKeys.BOXES: bboxes,
            OutputKeys.KEYPOINTS: lms,
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
