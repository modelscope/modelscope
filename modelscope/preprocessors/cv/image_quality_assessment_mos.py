# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from typing import Any, Dict, Union

import cv2
import numpy as np
from torchvision import transforms

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.cv,
    module_name=Preprocessors.image_quality_assessment_mos_preprocessor)
class ImageQualityAssessmentMosPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        """Preprocess the image for image quality assessment .
        """
        super().__init__(**kwargs)

    def preprocessors(self, input):
        img = LoadImage.convert_to_ndarray(input)
        sub_img_dim = (720, 1280)
        resize_dim = (1080, 1920)
        h, w = img.shape[:2]

        resize_h, resize_w = resize_dim
        sub_img_h, sub_img_w = sub_img_dim
        flag = False
        if (w - h) * (resize_w - resize_h) < 0:
            flag = True
            resize_w, resize_h = resize_h, resize_w
            sub_img_w, sub_img_h = sub_img_h, sub_img_w

        # 注意只能等比例缩放
        w_scale = resize_w / w
        h_scale = resize_h / h
        scale = max(h_scale, w_scale)
        img = cv2.resize(
            img, (int(math.ceil(scale * w)), int(math.ceil(scale * h))),
            interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        h_i = (h - sub_img_h) // 2
        w_i = (w - sub_img_w) // 2
        img = img[h_i:h_i + sub_img_h, w_i:w_i + sub_img_w, :]

        if flag:
            img = np.rot90(img)
        img = LoadImage.convert_to_img(img)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)
        return img

    @type_assert(object, object)
    def __call__(self, input) -> Dict[str, Any]:
        data = self.preprocessors(input)
        ret = {'input': data.unsqueeze(0)}
        return ret
