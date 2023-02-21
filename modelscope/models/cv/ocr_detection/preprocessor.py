# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor, load_image
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.ocr_detection)
class OCRDetectionPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, mode: str = ModeKeys.INFERENCE):
        """The base constructor for all ocr recognition preprocessors.

        Args:
            model_dir (str): model directory to initialize some resource
            mode: The mode for the preprocessor.
        """
        super().__init__(mode)
        cfgs = Config.from_file(
            os.path.join(model_dir, ModelFile.CONFIGURATION))
        self.image_short_side = cfgs.model.inference_kwargs.image_short_side

    def __call__(self, inputs):
        """process the raw input data
        Args:
            inputs:
                - A string containing an HTTP link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL or opencv directly
        Returns:
            outputs: the preprocessed image
        """
        if isinstance(inputs, str):
            img = np.array(load_image(inputs))
        elif isinstance(inputs, PIL.Image.Image):
            img = np.array(inputs)
        else:
            raise TypeError(
                f'inputs should be either str, PIL.Image, np.array, but got {type(inputs)}'
            )

        img = img[:, :, ::-1]
        height, width, _ = img.shape
        if height < width:
            new_height = self.image_short_side
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.image_short_side
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        resized_img = resized_img - np.array([123.68, 116.78, 103.94],
                                             dtype=np.float32)
        resized_img /= 255.
        resized_img = torch.from_numpy(resized_img).permute(
            2, 0, 1).float().unsqueeze(0)

        result = {'img': resized_img, 'org_shape': [height, width]}
        return result
