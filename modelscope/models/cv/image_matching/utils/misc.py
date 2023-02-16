# This implementation is adopted from LoFTR,
# made public available under the Apache License, Version 2.0,
# at https://github.com/zju3dv/LoFTR

from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}
