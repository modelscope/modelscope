# Copyright (c) Alibaba, Inc. and its affiliates.

import colorsys
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile, ImageFont

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.cv.tbs_detection_utils.utils import (_get_anchors,
                                                               generate,
                                                               post_process)
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger()

__all__ = ['TBSDetectionPipeline']


@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.tbs_detection)
class TBSDetectionPipeline(Pipeline):
    """ TBS Detection Pipeline.

    Example:

    ```python
    >>> from modelscope.pipelines import pipeline

    >>> tbs_detect = pipeline(Tasks.image_object_detection, model='landingAI/LD_CytoBrainCerv')
    >>> tbs_detect(input='data/test/images/tbs_detection.jpg')
       {
        "boxes": [
            [
            446.9007568359375,
            36.374977111816406,
            907.0919189453125,
            337.439208984375
            ],
            [
            454.3310241699219,
            336.08477783203125,
            921.26904296875,
            641.7871704101562
            ]
        ],
        "labels": [
            ["Positive"]
        ],
        "scores": [
            0.9296008944511414,
            0.9260380268096924
        ]
        }
    >>> #
    ```
    """
    _defaults = {
        'class_names': ['positive'],
        'model_image_size': (416, 416, 3),
        'confidence': 0.5,
        'iou': 0.3,
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
        """
        Detect objects (bounding boxes) in the image(s) passed as inputs.

        Args:
            input (`Image` or `List[Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL or opencv directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format.


        Return:
            A dictionary of result or a list of dictionary of result. If the input is an image, a dictionary
            is returned. If input is a list of image, a list of dictionary is returned.

            The dictionary contain the following keys:

            - **scores** (`List[float]`) -- The detection score for each card in the image.
            - **boxes** (`List[float]) -- The bounding boxe [x1, y1, x2, y2] of detected objects in in image's
                original size.
            - **labels** (`List[str]`, optional) -- The boxes's class_names of detected object in image.
        """
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
        result = {'data': outputs, 'img_path': input['img_path']}
        return result

    def postprocess(self, input: Dict[str, Any], *args,
                    **kwargs) -> Dict[str, Any]:

        bboxes, scores = post_process(self, input['data'], input['img_path'])

        if bboxes is None:
            outputs = {OutputKeys.SCORES: [], OutputKeys.BOXES: []}
            return outputs
        outputs = {
            OutputKeys.SCORES: scores.tolist(),
            OutputKeys.LABELS: ['Positive'],
            OutputKeys.BOXES: bboxes
        }
        return outputs
