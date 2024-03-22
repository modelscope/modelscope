# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import cv2
import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.image_driving_perception_preprocessor)
class ImageDrivingPerceptionPreprocessor(Preprocessor):

    def __init__(self, mode: str = ModeKeys.INFERENCE, *args, **kwargs):
        """
        Args:
            model_dir (str): model directory to initialize some resource.
            mode: The mode for the preprocessor.
        """
        super().__init__(mode, *args, **kwargs)

    def _check_image(self, input_img):
        whole_temp_shape = input_img.shape
        if len(whole_temp_shape) == 2:
            input_img = np.stack([input_img, input_img, input_img], axis=2)
        elif whole_temp_shape[2] == 1:
            input_img = np.concatenate([input_img, input_img, input_img],
                                       axis=2)
        elif whole_temp_shape[2] == 4:
            input_img = input_img[:, :,
                                  0:3] * 1.0 * input_img[:, :,
                                                         3:4] * 1.0 / 255.0
        return input_img

    def _letterbox(self,
                   img,
                   new_shape=(640, 640),
                   color=(114, 114, 114),
                   auto=True,
                   scaleFill=False,
                   scaleup=True,
                   stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)  # add border

        return img, ratio, (dw, dh)

    @type_assert(object, object)
    def __call__(
        self, data: str, output_shape=(1280, 720), new_shape=(640, 640)
    ) -> Dict[str, Any]:
        """process the raw input data
        Args:
            data (str): image path
        Returns:
            Dict[ndarray, Any]: the preprocessed data
            {
                "img": the preprocessed resized image (640x640)
            }
        """
        img = LoadImage.convert_to_ndarray(data)
        if img is not None:
            img = self._check_image(img)
        else:
            raise Exception('img is None')
        ori_h, ori_w, _ = img.shape
        img = self._letterbox(img, new_shape)[0]
        img = img.transpose(2, 0, 1)  # to 3x640x640

        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        # Convert
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return {
            'img': img,
            'ori_img_shape': (ori_h, ori_w),
        }
