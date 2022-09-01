# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

import os

from ..yolox_base import Exp as YoloXExp


class YoloXTinyExp(YoloXExp):

    def __init__(self):
        super(YoloXTinyExp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.enable_mixup = False
