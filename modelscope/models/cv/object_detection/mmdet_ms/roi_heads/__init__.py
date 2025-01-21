# Implementation in this file is modified based on ViTAE-Transformer
# Originally Apache 2.0 License and publicly available at https://github.com/ViTAE-Transformer/ViTDet
from .bbox_heads import (ConvFCBBoxNHead, Shared2FCBBoxNHead,
                         Shared4Conv1FCBBoxNHead)
from .mask_heads import FCNMaskNHead

__all__ = [
    'ConvFCBBoxNHead', 'Shared2FCBBoxNHead', 'Shared4Conv1FCBBoxNHead',
    'FCNMaskNHead'
]
