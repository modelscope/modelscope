"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/datasets/pipelines
"""
from .auto_augment import RotateV2
from .formating import DefaultFormatBundleV2
from .loading import LoadAnnotationsV2
from .transforms import RandomSquareCrop

__all__ = [
    'RandomSquareCrop', 'LoadAnnotationsV2', 'RotateV2',
    'DefaultFormatBundleV2'
]
