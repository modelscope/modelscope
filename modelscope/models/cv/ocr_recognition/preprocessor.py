# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os

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
    Fields.cv, module_name=Preprocessors.ocr_recognition)
class OCRRecognitionPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, mode: str = ModeKeys.INFERENCE):
        """The base constructor for all ocr recognition preprocessors.

        Args:
            model_dir (str): model directory to initialize some resource
            mode: The mode for the preprocessor.
        """
        super().__init__(mode)
        cfgs = Config.from_file(
            os.path.join(model_dir, ModelFile.CONFIGURATION))
        self.do_chunking = cfgs.model.inference_kwargs.do_chunking
        self.target_height = cfgs.model.inference_kwargs.img_height
        self.target_width = cfgs.model.inference_kwargs.img_width

    def keepratio_resize(self, img):
        cur_ratio = img.shape[1] / float(img.shape[0])
        mask_height = self.target_height
        mask_width = self.target_width
        if cur_ratio > float(self.target_width) / self.target_height:
            cur_target_height = self.target_height
            cur_target_width = self.target_width
        else:
            cur_target_height = self.target_height
            cur_target_width = int(self.target_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[:img.shape[0], :img.shape[1]] = img
        img = mask
        return img

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
            img = np.array(load_image(inputs).convert('L'))
        elif isinstance(inputs, PIL.Image.Image):
            img = np.array(inputs.convert('L'))
        elif isinstance(inputs, np.ndarray):
            if len(inputs.shape) == 3:
                img = cv2.cvtColor(inputs, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(
                f'inputs should be either str, PIL.Image, np.array, but got {type(inputs)}'
            )

        if self.do_chunking:
            PRED_LENTH = 75
            PRED_PAD = 6
            data = []
            img_h, img_w = img.shape
            wh_ratio = img_w / img_h
            true_w = int(self.target_height * wh_ratio)
            split_batch_cnt = 1
            if true_w < self.target_width * 1.2:
                img = cv2.resize(
                    img, (min(true_w, self.target_width), self.target_height))
            else:
                split_batch_cnt = math.ceil((true_w - 48) * 1.0 / 252)
                img = cv2.resize(img, (true_w, self.target_height))

            if split_batch_cnt == 1:
                mask = np.zeros((self.target_height, self.target_width))
                mask[:, :img.shape[1]] = img
                data.append(mask)
            else:
                for idx in range(split_batch_cnt):
                    mask = np.zeros((self.target_height, self.target_width))
                    left = (PRED_LENTH * 4 - PRED_PAD * 4) * idx
                    trunk_img = img[:, left:min(left + PRED_LENTH * 4, true_w)]
                    mask[:, :trunk_img.shape[1]] = trunk_img
                    data.append(mask)

            data = torch.FloatTensor(data).view(
                len(data), 1, self.target_height, self.target_width) / 255.
        else:
            data = self.keepratio_resize(img)
            data = torch.FloatTensor(data).view(1, 1, self.target_height,
                                                self.target_width) / 255.
        return data
