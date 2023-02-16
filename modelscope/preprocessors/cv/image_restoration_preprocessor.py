# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from typing import Any, Dict

import torch
import torch.nn.functional as F
from numpy import ndarray
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.image_demoire_preprocessor)
class ImageRestorationPreprocessor(Preprocessor):

    def __init__(self, pad_32, min_max_l, **kwargs):
        super().__init__(**kwargs)

        self.pad_32 = pad_32
        self.min_max_l = min_max_l
        self.transform_input = transforms.Compose([transforms.ToTensor()])

    def img_pad_3(self, x, w_pad, h_pad, w_odd_pad, h_odd_pad):
        x1 = F.pad(
            x[0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3827)
        x2 = F.pad(
            x[1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.4141)
        x3 = F.pad(
            x[2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3912)
        y = torch.cat([x1, x2, x3], dim=0)
        return y

    @type_assert(object, object)
    def __call__(self, data: ndarray) -> Dict[str, Any]:
        image = Image.fromarray(data)
        img_w, img_h = image.size
        min_wh = min(img_w, img_h)
        # set min_max_l is 3072 avoid gpu oom(16G)
        if min_wh > self.min_max_l:
            fscale = self.min_max_l / min_wh
            img_w_n = int(img_w * fscale)
            img_h_n = int(img_h * fscale)
            img_w_n = math.ceil(img_w_n / 32) * 32
            img_h_n = math.ceil(img_h_n / 32) * 32
            image = image.resize((img_w_n, img_h_n))
        data = self.transform_input(image)
        h_pad = 0
        h_odd_pad = 0
        w_pad = 0
        w_odd_pad = 0
        if self.pad_32:
            c, h, w = data.size()
            # pad image such that the resolution is a multiple of 32
            w_pad = (math.ceil(w / 32) * 32 - w) // 2
            h_pad = (math.ceil(h / 32) * 32 - h) // 2
            w_odd_pad = w_pad
            h_odd_pad = h_pad
            if w % 2 == 1:
                w_odd_pad += 1
            if h % 2 == 1:
                h_odd_pad += 1
            data = self.img_pad_3(
                data,
                w_pad=w_pad,
                h_pad=h_pad,
                w_odd_pad=w_odd_pad,
                h_odd_pad=h_odd_pad)

        data = data.unsqueeze(0)
        return {
            'img': data.float(),
            'h_pad': h_pad,
            'h_odd_pad': h_odd_pad,
            'w_pad': w_pad,
            'w_odd_pad': w_odd_pad
        }
