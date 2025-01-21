# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class Exp(BaseExp):

    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 80
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = 'silu'
        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

    def get_model(self):
        from ..models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, 'model', None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
