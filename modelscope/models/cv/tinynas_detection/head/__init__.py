# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.

import copy

from .gfocal_v2_tiny import GFocalHead_Tiny
from .zero_head import ZeroHead


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'GFocalV2':
        return GFocalHead_Tiny(**head_cfg)
    elif name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    else:
        raise NotImplementedError
