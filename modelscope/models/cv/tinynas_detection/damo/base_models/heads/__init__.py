# Copyright (c) Alibaba, Inc. and its affiliates.

import copy

from .gfocal_v2_tiny import GFocalHead_Tiny
from .zero_head import ZeroHead


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    elif name == 'GFocalV2':
        return GFocalHead_Tiny(**head_cfg)
    else:
        raise NotImplementedError
