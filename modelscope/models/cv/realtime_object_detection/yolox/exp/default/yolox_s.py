# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

import os

from ..yolox_base import Exp as YoloXExp


class YoloXSExp(YoloXExp):

    def __init__(self):
        super(YoloXSExp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
