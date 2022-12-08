"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/models/detectors
"""
from .scrfd import SCRFD
from .tinymog import TinyMog

__all__ = ['SCRFD', 'TinyMog']
