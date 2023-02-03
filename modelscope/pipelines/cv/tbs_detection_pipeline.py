# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict
import torch
import numpy as np
import cv2
import os
import colorsys
from PIL import ImageFile
from PIL import Image, ImageFont, ImageDraw
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.pipelines.cv.tbs_detection_utils.utils import _get_anchors, generate, post_process


ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger()

__all__ = ['TBSDetectionPipeline']

@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.tbs_detection)
class TBSDetectionPipeline(Pipeline):

    _defaults = {
        "class_names": ['positive'],
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "iou": 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        self.__dict__.update(self._defaults)
        self.anchors = _get_anchors(self)
        generate(self)


    def preprocess(self, input: Input) -> Dict[str, Any]:

        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float)
        result = {'img': img, 'img_path': input}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        img = input['img'].astype(np.uint8)
        img = cv2.resize(img, (416, 416))
        img = img.astype(np.float32)
        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_inp = torch.from_numpy(tmp_inp).type(torch.FloatTensor)
        img = torch.unsqueeze(tmp_inp, dim=0)
        model_path = os.path.join(self.model, 'pytorch_yolov4.pt')
        model = torch.load(model_path)
        outputs = model(img.cuda())
        result = {'data': outputs ,'img_path': input['img_path']}
        return result

    def postprocess(self, input: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:

        bboxes, scores = post_process(self, input['data'], input['img_path'])

        if bboxes is None:
            outputs = {
                OutputKeys.SCORES: [],
                OutputKeys.BOXES: []
            }
            return outputs
        outputs = {
            OutputKeys.SCORES: scores.tolist(),
            OutputKeys.LABELS: ["Positive"],
            OutputKeys.BOXES: bboxes
        }
        return outputs

