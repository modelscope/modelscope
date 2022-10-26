# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .networks import StaticGestureNet

logger = get_logger()

map_idx = {
    0: 'unrecog',
    1: 'one',
    2: 'two',
    3: 'bixin',
    4: 'yaogun',
    5: 'zan',
    6: 'fist',
    7: 'ok',
    8: 'tuoju',
    9: 'd_bixin',
    10: 'd_fist_left',
    11: 'd_fist_right',
    12: 'd_hand',
    13: 'fashe',
    14: 'five',
    15: 'nohand'
}

img_size = [112, 112]

spatial_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@MODELS.register_module(Tasks.hand_static, module_name=Models.hand_static)
class HandStatic(TorchModel):

    def __init__(self, model_dir, device_id=0, *args, **kwargs):

        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)

        self.model = StaticGestureNet()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.params = torch.load(
            '{}/{}'.format(model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location=self.device)

        self.model.load_state_dict(self.params)
        self.model.to(self.device)
        self.model.eval()
        self.device_id = device_id
        if self.device_id >= 0 and self.device == 'cuda':
            self.model.to('cuda:{}'.format(self.device_id))
            logger.info('Use GPU: {}'.format(self.device_id))
        else:
            self.device_id = -1
            logger.info('Use CPU for inference')

    def forward(self, x):
        pred_result = self.model(x)
        return pred_result


def infer(img, model, device):
    img = img.cpu().numpy()
    img = Image.fromarray(img)
    clip = spatial_transform(img)
    clip = clip.unsqueeze(0).to(device).float()
    outputs = model(clip)
    predicted = int(outputs.max(1)[1])
    pred_result = map_idx.get(predicted)
    logger.info('pred result: {}'.format(pred_result))

    return pred_result
