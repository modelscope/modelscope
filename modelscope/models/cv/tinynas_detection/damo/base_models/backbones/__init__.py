# Copyright (c) Alibaba, Inc. and its affiliates.

import copy

from .darknet import CSPDarknet
from .tinynas_csp import load_tinynas_net as load_tinynas_net_csp
from .tinynas_res import load_tinynas_net as load_tinynas_net_res


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'TinyNAS_res':
        return load_tinynas_net_res(backbone_cfg)
    elif name == 'TinyNAS_csp':
        return load_tinynas_net_csp(backbone_cfg)
    elif name == 'CSPDarknet':
        return CSPDarknet(**backbone_cfg)
    else:
        print(f'{name} is not supported yet!')
