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
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from . import FaceProcessingBasePipeline

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_liveness, module_name=Pipelines.face_liveness_xc)
class FaceLivenessXcPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        FaceLivenessXcPipeline can judge the input face is a real or fake face.
        use `model` to create a face lievness ir pipeline for prediction
        Args:
            model: model id on modelscope hub.
        ```python
        >>> from modelscope.pipelines import pipeline
        >>> fl_xc = pipeline('face_liveness', 'damo/cv_manual_face-liveness_flxc')
        >>> fl_xc("https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_liveness_xc.png")
        {'scores': [0.03821974992752075], 'boxes': [[12.569677352905273, 6.428711891174316,
            94.17887115478516, 106.74441528320312]]}
        ```
        """
        super().__init__(model=model, **kwargs)
        onnx_path = osp.join(model, ModelFile.ONNX_MODEL_FILE)
        logger.info(f'loading model from {onnx_path}')
        self.sess, self.input_node_name, self.out_node_name = self.load_onnx_model(
            onnx_path)
        logger.info('load model done')

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
        img = result['img']
        img = (img - 127.5) * 0.0078125
        img = np.expand_dims(img, 0).copy()
        input_tensor = np.concatenate([img, img, img, img], axis=3)
        input_tensor = np.transpose(
            input_tensor, axes=(0, 3, 1, 2)).astype(np.float32)
        result['input_tensor'] = input_tensor
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        input_feed = {}
        input_feed[
            self.input_node_name[0]] = input['input_tensor'].cpu().numpy()
        result = self.sess.run(self.out_node_name, input_feed=input_feed)
        scores = [result[0][0][0].tolist()]

        boxes = input['bbox'].cpu().numpy()[np.newaxis, :].tolist()
        return {OutputKeys.SCORES: scores, OutputKeys.BOXES: boxes}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
