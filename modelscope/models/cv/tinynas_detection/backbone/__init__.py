# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

import copy

from .darknet import CSPDarknet
from .tinynas import load_tinynas_net


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'CSPDarknet':
        return CSPDarknet(**backbone_cfg)
    elif name == 'TinyNAS':
        return load_tinynas_net(backbone_cfg)
