# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

import copy

from .giraffe_fpn import GiraffeNeck
from .giraffe_fpn_v2 import GiraffeNeckV2


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'GiraffeNeck':
        return GiraffeNeck(**neck_cfg)
    elif name == 'GiraffeNeckV2':
        return GiraffeNeckV2(**neck_cfg)
