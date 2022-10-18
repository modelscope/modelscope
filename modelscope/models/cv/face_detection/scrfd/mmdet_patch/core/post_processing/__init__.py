"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/core/post_processing/bbox_nms.py
"""
from .bbox_nms import multiclass_nms

__all__ = ['multiclass_nms']
