# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import cv2
import numpy as np
import torch
from PIL import Image

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .net import F3Net

logger = get_logger()


def load_state_dict(model_dir, device):
    _dict = torch.load(
        '{}/{}'.format(model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
        map_location=device)
    state_dict = {}
    for k, v in _dict.items():
        if k.startswith('module'):
            k = k[7:]
        state_dict[k] = v
    return state_dict


@MODELS.register_module(
    Tasks.product_segmentation, module_name=Models.product_segmentation)
class F3NetForProductSegmentation(TorchModel):

    def __init__(self, model_dir, device_id=0, *args, **kwargs):

        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)

        self.model = F3Net()
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')

        self.params = load_state_dict(model_dir, self.device)
        self.model.load_state_dict(self.params)
        self.model.to(self.device)
        self.model.eval()
        self.model.to(self.device)

    def forward(self, x):
        pred_result = self.model(x)
        return pred_result


mean, std = np.array([[[124.55, 118.90,
                        102.94]]]), np.array([[[56.77, 55.97, 57.50]]])


def inference(model, device, img):
    img = img.cpu().numpy()
    img = (img - mean) / std
    img = cv2.resize(img, dsize=(448, 448), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.to(device).float()
    outputs = model(img)
    out = outputs[0]
    pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
    pred[pred < 20] = 0
    pred = pred[:, :, np.newaxis]
    pred = np.round(pred)
    logger.info('Inference Done')
    return pred
