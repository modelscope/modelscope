# Implementation in this file is modified based on ViTAE-Transformer
# Originally Apache 2.0 License and publicly available at https://github.com/ViTAE-Transformer/ViTDet
from .convfc_bbox_head import (ConvFCBBoxNHead, Shared2FCBBoxNHead,
                               Shared4Conv1FCBBoxNHead)

__all__ = ['ConvFCBBoxNHead', 'Shared2FCBBoxNHead', 'Shared4Conv1FCBBoxNHead']
