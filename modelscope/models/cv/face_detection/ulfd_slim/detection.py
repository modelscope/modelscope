# The implementation is based on ULFD, available at
# https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .vision.ssd.fd_config import define_img_size
from .vision.ssd.mb_tiny_fd import (create_mb_tiny_fd,
                                    create_mb_tiny_fd_predictor)

define_img_size(640)


@MODELS.register_module(Tasks.face_detection, module_name=Models.ulfd)
class UlfdFaceDetector(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.net = create_mb_tiny_fd(2, is_test=True, device=device)
        self.predictor = create_mb_tiny_fd_predictor(
            self.net, candidate_size=1500, device=device)
        self.net.load(model_path)
        self.net = self.net.to(device)

    def forward(self, input):
        img_raw = input['img']
        img = np.array(img_raw.cpu().detach())
        img = img[:, :, ::-1]
        prob_th = 0.85
        keep_top_k = 750
        boxes, labels, probs = self.predictor.predict(img, keep_top_k, prob_th)
        return boxes, probs
