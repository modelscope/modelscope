# Copyright (c) Alibaba, Inc. and its affiliates.

from torch.nn.parallel.distributed import DistributedDataParallel

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg

PARALLEL = Registry('parallel')
PARALLEL.register_module(
    module_name='DistributedDataParallel', module_cls=DistributedDataParallel)


def build_parallel(cfg: ConfigDict, default_args: dict = None):
    """ build parallel

    Args:
        cfg (:obj:`ConfigDict`): config dict for parallel object.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(cfg, PARALLEL, default_args=default_args)
