# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import onnxruntime
import PIL
import torch
import torch.nn.functional as F

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.models.cv.facial_landmark_confidence import \
    FacialLandmarkConfidence
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from . import FaceProcessingBasePipeline

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_quality_assessment,
    module_name=Pipelines.face_quality_assessment)
class FaceQualityAssessmentPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face quality assessment pipeline for prediction
        Args:
            model: model id on modelscope hub.
        Example:
        FaceQualityAssessmentPipeline can measure the quality of an input face image,
        the higher output score represents the better quality

        ```python
        >>> from modelscope.pipelines import pipeline
        >>> fqa = pipeline('face-quality-assessment', 'damo/cv_manual_face-quality-assessment_fqa')
        >>> frfm("https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png")
        {'scores': [0.99949193], 'boxes': [[157.72341918945312, 67.5608139038086,
            305.8574523925781, 271.25555419921875]]}

        ```
        """
        super().__init__(model=model, **kwargs)
        onnx_path = osp.join(model, ModelFile.ONNX_MODEL_FILE)
        logger.info(f'loading model from {onnx_path}')
        self.sess, self.input_node_name, self.out_node_name = self.load_onnx_model(
            onnx_path)
        logger.info('load model done')

    def _batch(self, data):
        return batch_process(self.model, data)

    def load_onnx_model(self, onnx_path):
        sess = onnxruntime.InferenceSession(onnx_path)
        out_node_name = []
        input_node_name = []
        for node in sess.get_outputs():
            out_node_name.append(node.name)

        for node in sess.get_inputs():
            input_node_name.append(node.name)

        return sess, input_node_name, out_node_name

    def preprocess(self, input: Input) -> Dict[str, Any]:
        result = super().preprocess(input)
        align_img = result['img']
        face_img = align_img[:, :, ::-1]  # to rgb
        face_img = (face_img / 255. - 0.5) / 0.5
        face_img = np.expand_dims(face_img, 0).copy()
        face_img = np.transpose(face_img, axes=(0, 3, 1, 2))
        face_img = face_img.astype(np.float32)
        result['input_tensor'] = face_img
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        input_feed = {}
        input_feed[
            self.input_node_name[0]] = input['input_tensor'].cpu().numpy()
        result = self.sess.run(self.out_node_name, input_feed=input_feed)
        assert result is not None
        scores = [result[0][0][0]]
        boxes = input['bbox'].cpu().numpy()[np.newaxis, :].tolist()
        return {OutputKeys.SCORES: scores, OutputKeys.BOXES: boxes}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
