# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os.path as osp
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.table_recognition import LoreModel
from modelscope.models.cv.table_recognition.lineless_table_process import \
    get_affine_transform_upper_left
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import load_image
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.lineless_table_recognition,
    module_name=Pipelines.lineless_table_recognition)
class LinelessTableRecognitionPipeline(Pipeline):
    r""" Lineless Table Recognition Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> detector = pipeline('lineless-table-recognition', 'damo/cv_resnet-transformer_table-structure-recognition_lore')
    >>> detector("data/test/images/lineless_table_recognition.jpg")
    >>>   {
    >>>    "polygons": [
    >>>        [
    >>>            159.65718,
    >>>            161.14981,
    >>>            170.9718,
    >>>            161.1621,
    >>>            170.97322,
    >>>            175.4334,
    >>>            159.65717,
    >>>            175.43259
    >>>        ],
    >>>        [
    >>>            153.24953,
    >>>            230.49915,
    >>>            176.26964,
    >>>            230.50377,
    >>>            176.26273,
    >>>            246.08868,
    >>>            153.24817,
    >>>            246.10458
    >>>        ],
    >>>        ......
    >>>    ],
    >>>    "boxes": [
    >>>        [
    >>>            4.,
    >>>            4.,
    >>>            1.,
    >>>            1.
    >>>        ],
    >>>        [
    >>>            6.,
    >>>            6.,
    >>>            1.,
    >>>            1.
    >>>        ],
    >>>        ......
    >>>    ]
    >>>   }
    >>>
    """

    def __init__(self, model: Union[Model, str], **kwargs):
        """
        Args:
            model: model id on modelscope hub.
        """
        assert isinstance(model, str), 'model must be a single str'
        super().__init__(model=model, **kwargs)
        logger.info(f'loading model from dir {model}')
        self.model.eval()

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)[:, :, ::-1]

        mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]
        inp_height, inp_width = 768, 768
        c = np.array([0, 0], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform_upper_left(c, s, 0,
                                                      [inp_width, inp_height])

        resized_image = cv2.resize(img, (width, height))
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height,
                                                      inp_width)
        images = torch.from_numpy(images).to(self.device)
        meta = {
            'c': c,
            's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4
        }

        result = {'img': images, 'meta': meta}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
