"""
The implementation here is modified based on insightface, originally MIT license and publicly avaialbe at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/models/backbones
"""
from .mobilenet import MobileNetV1
from .resnet import ResNetV1e

__all__ = ['ResNetV1e', 'MobileNetV1']
