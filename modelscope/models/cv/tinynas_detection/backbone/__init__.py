# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.

import copy

from .darknet import CSPDarknet
from .tinynas_csp import load_tinynas_net as load_tinynas_net_csp
from .tinynas_res import load_tinynas_net as load_tinynas_net_res


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'CSPDarknet':
        return CSPDarknet(**backbone_cfg)
    elif name == 'TinyNAS_csp':
        return load_tinynas_net_csp(backbone_cfg)
    elif name == 'TinyNAS_res':
        return load_tinynas_net_res(backbone_cfg)
