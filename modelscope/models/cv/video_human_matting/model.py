# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Optional

import numpy as np
import torch
import torchvision
from torch.nn import functional as F

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_human_matting.models import MattingNetwork
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger


@MODELS.register_module(
    Tasks.video_human_matting, module_name=Models.video_human_matting)
class VideoMattingNetwork(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        params = torch.load(model_path, map_location='cpu')
        self.model = MattingNetwork()
        if 'model_state_dict' in params.keys():
            params = params['model_state_dict']
        self.model.load_state_dict(params, strict=True)
        self.model.eval()


def preprocess(image):
    frame_np = np.float32(image) / 255.0
    frame_np = frame_np.transpose(2, 0, 1)
    frame_tensor = torch.from_numpy(frame_np)
    image_tensor = frame_tensor[None, :, :, :]
    return image_tensor
