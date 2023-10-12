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
    Tasks.face_liveness, module_name=Pipelines.face_liveness_ir)
class FaceLivenessIrPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face lievness ir pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        onnx_path = osp.join(model, ModelFile.ONNX_MODEL_FILE)
        logger.info(f'loading model from {onnx_path}')
        self.sess, self.input_node_name, self.out_node_name = self.load_onnx_model(
            onnx_path)
        logger.info('load model done')

    def load_onnx_model(self, onnx_path):
        sess = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        out_node_name = []
        input_node_name = []
        for node in sess.get_outputs():
            out_node_name.append(node.name)

        for node in sess.get_inputs():
            input_node_name.append(node.name)

        return sess, input_node_name, out_node_name

    def preprocess(self, input: Input) -> Dict[str, Any]:

        result = super().preprocess(input)
        if result is None:
            rtn_dict = {}
            rtn_dict['input_tensor'] = None
            return rtn_dict
        orig_img = LoadImage.convert_to_ndarray(input)
        orig_img = orig_img[:, :, ::-1]
        img = super(FaceLivenessIrPipeline,
                    self).align_face_padding(orig_img, result['bbox'], 16)
        if img.shape[0] != 112:
            img = img[8:120, 8:120, :]
        img = (img - 127.5) * 0.0078125
        input_tensor = img.astype('float32').transpose(
            (2, 0, 1))[np.newaxis, :]
        result['input_tensor'] = input_tensor
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if input['input_tensor'] is None:
            return {OutputKeys.SCORES: None, OutputKeys.BOXES: None}
        input_feed = {}
        input_feed[
            self.input_node_name[0]] = input['input_tensor'].cpu().numpy()
        result = self.sess.run(self.out_node_name, input_feed=input_feed)
        out = F.softmax(torch.FloatTensor(result), dim=-1)[0][0]
        assert result is not None
        scores = [1 - out[1].tolist()]
        boxes = input['bbox'].cpu().numpy()[np.newaxis, :].tolist()
        return {OutputKeys.SCORES: scores, OutputKeys.BOXES: boxes}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
