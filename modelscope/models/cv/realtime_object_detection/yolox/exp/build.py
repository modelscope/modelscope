# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

import os
import sys


def get_exp_by_name(exp_name):
    exp = exp_name.replace('-',
                           '_')  # convert string like "yolox-s" to "yolox_s"
    if exp == 'yolox_s':
        from .default import YoloXSExp as YoloXExp
    elif exp == 'yolox_nano':
        from .default import YoloXNanoExp as YoloXExp
    elif exp == 'yolox_tiny':
        from .default import YoloXTinyExp as YoloXExp
    elif exp == 'streamyolo':
        from .default import StreamYoloExp as YoloXExp
    else:
        pass
    return YoloXExp()
