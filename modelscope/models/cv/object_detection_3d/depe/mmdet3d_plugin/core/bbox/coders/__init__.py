"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/core/bbox/coders
"""
from .nms_free_coder import NMSFreeCoder

__all__ = [
    'NMSFreeCoder',
]
