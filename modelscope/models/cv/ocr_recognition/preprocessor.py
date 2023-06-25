# Copyright (c) Alibaba, Inc. and its affiliates.

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
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[:img.shape[0], :img.shape[1], :] = img
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
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            if isinstance(item, str):
                img = np.array(load_image(item).convert('RGB'))
            elif isinstance(item, PIL.Image.Image):
                img = np.array(item.convert('RGB'))
            elif isinstance(item, np.ndarray):
                if len(item.shape) == 2:
                    img = cv2.cvtColor(item, cv2.COLOR_GRAY2RGB)
                else:
                    img = item
            else:
                raise TypeError(
                    f'inputs should be either (a list of) str, PIL.Image, np.array, but got {type(item)}'
                )
            img = self.keepratio_resize(img)
            img = torch.FloatTensor(img)
            if self.do_chunking:
                chunk_img = []
                for i in range(3):
                    left = (300 - 48) * i
                    chunk_img.append(img[:, left:left + 300, :])
                merge_img = torch.cat(chunk_img, 0)
                data = merge_img.view(3, self.target_height, 300, 3) / 255.
                data = data.permute(0, 3, 1, 2)
            else:
                data = img.view(1, self.target_height, self.target_width,
                                3) / 255.
                data = data.permute(0, 3, 1, 2)
            data_batch.append(data)
        data_batch = torch.cat(data_batch, 0)
        return {'image': data_batch}
