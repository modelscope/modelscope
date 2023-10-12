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
    Tasks.face_recognition, module_name=Pipelines.face_recognition_onnx_fm)
class FaceRecognitionOnnxFmPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        FaceRecognitionOnnxFmPipeline can extract 512-dim feature of mask or non-masked face image. use `model`
        to create a face recognition face mask onnx pipeline for prediction.

        Args:
            model: model id on modelscope hub.

        Examples:

        >>> from modelscope.pipelines import pipeline
        >>> frfm = pipeline('face-recognition-ood', 'damo/cv_manual_face-recognition_frfm')
        >>> frfm("https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png")
        >>> {{'img_embedding': array([[ 0.02276129, -0.00761525, ...,0.05735306]],
        >>>    dtype=float32)} }
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
        align_img = result['img']
        face_img = align_img[:, :, ::-1]  # to rgb
        face_img = (face_img / 255. - 0.5) / 0.5
        face_img = np.expand_dims(face_img, 0).copy()
        face_img = np.transpose(face_img, axes=(0, 3, 1, 2))
        face_img = face_img.astype(np.float32)
        result['input_tensor'] = face_img
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if input['input_tensor'] is None:
            return {OutputKeys.IMG_EMBEDDING: None}
        input_feed = {}
        input_feed[
            self.input_node_name[0]] = input['input_tensor'].cpu().numpy()
        emb = self.sess.run(self.out_node_name, input_feed=input_feed)[0]
        emb /= np.sqrt(np.sum(emb**2, -1, keepdims=True))  # l2 norm
        return {OutputKeys.IMG_EMBEDDING: emb}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
