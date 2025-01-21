# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.models.cv.face_recognition.torchkit.backbone.arcface_backbone import \
    _iresnet
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
    Tasks.face_recognition, module_name=Pipelines.arc_face_recognition)
class ArcFaceRecognitionPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        # face recong model
        super().__init__(model=model, **kwargs)
        face_model = _iresnet('arcface_i50', [3, 4, 14, 3])
        face_model.load_state_dict(
            torch.load(
                osp.join(model, ModelFile.TORCH_MODEL_FILE),
                map_location=self.device))
        face_model = face_model.to(self.device)
        face_model.eval()
        self.face_model = face_model
        logger.info('face recognition model loaded!')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        result = super().preprocess(input)
        if result is None:
            rtn_dict = {}
            rtn_dict['img'] = None
            return rtn_dict
        align_img = result['img']
        face_img = align_img[:, :, ::-1]  # to rgb
        face_img = np.transpose(face_img, axes=(2, 0, 1))
        face_img = (face_img / 255. - 0.5) / 0.5
        face_img = face_img.astype(np.float32)
        result['img'] = face_img
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if input['img'] is None:
            return {OutputKeys.IMG_EMBEDDING: None}
        img = input['img'].unsqueeze(0)
        emb = self.face_model(img).detach().cpu().numpy()
        emb /= np.sqrt(np.sum(emb**2, -1, keepdims=True))  # l2 norm
        return {OutputKeys.IMG_EMBEDDING: emb}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
