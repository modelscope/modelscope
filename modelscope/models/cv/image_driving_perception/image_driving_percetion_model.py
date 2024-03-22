# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['YOLOPv2']


@MODELS.register_module(
    Tasks.image_driving_perception, module_name=Models.yolopv2)
class YOLOPv2(TorchModel):
    """ YOLOPv2 use E-ELAN which first adopted in Yolov7 as backbone, SPP+FPN+PAN as neck and head.
    For more information, please refer to https://arxiv.org/pdf/2208.11434.pdf
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)

        self.model_dir = model_dir
        self._load_pretrained_checkpoint()

    def forward(self, data):
        img = data['img']
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img)
        return {
            'img_hw': data['img'].shape[2:],
            'ori_img_shape': data['ori_img_shape'],
            'pred': pred,
            'anchor_grid': anchor_grid,
            'driving_area_mask': seg,
            'lane_line_mask': ll,
        }

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return super().postprocess(inputs, **kwargs)

    def _load_pretrained_checkpoint(self):
        model_path = os.path.join(self.model_dir, ModelFile.TORCH_MODEL_FILE)
        logger.info(model_path)
        if os.path.exists(model_path):
            self.model = torch.jit.load(model_path, 'cpu')
            self.model = self.model.eval()

        else:
            logger.error(
                '[checkModelPath]:model path dose not exits!!! model Path:'
                + model_path)
            raise Exception('[checkModelPath]:model path dose not exits!')
