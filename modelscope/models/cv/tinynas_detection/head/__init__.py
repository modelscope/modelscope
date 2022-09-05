# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

import copy

from .gfocal_v2_tiny import GFocalHead_Tiny


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'GFocalV2':
        return GFocalHead_Tiny(**head_cfg)
    else:
        raise NotImplementedError
