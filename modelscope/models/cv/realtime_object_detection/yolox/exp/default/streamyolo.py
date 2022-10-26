# The implementation is based on StreamYOLO, available at https://github.com/yancie-yjr/StreamYOLO
import os
import sys

import torch

from ..yolox_base import Exp as YoloXExp


class StreamYoloExp(YoloXExp):

    def __init__(self):
        super(YoloXExp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.num_classes = 8
        self.test_size = (600, 960)
        self.test_conf = 0.3
        self.nmsthre = 0.65

    def get_model(self):
        from ...models import StreamYOLO, DFPPAFPN, TALHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, 'model', None) is None:
            in_channels = [256, 512, 1024]
            backbone = DFPPAFPN(
                self.depth, self.width, in_channels=in_channels)
            head = TALHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                gamma=1.0,
                ignore_thr=0.5,
                ignore_value=1.6)
            self.model = StreamYOLO(backbone, head)

        return self.model
