# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect

import torch

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg, default_group

OPTIMIZERS = Registry('optimizer')


def build_optimizer(model: torch.nn.Module,
                    cfg: ConfigDict,
                    default_args: dict = None):
    """ build optimizer from optimizer config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for optimizer object.
        default_args (dict, optional): Default initialization arguments.
    """
    if hasattr(model, 'module'):
        model = model.module

    if default_args is None:
        default_args = {}
    default_args['params'] = model.parameters()

    return build_from_cfg(
        cfg, OPTIMIZERS, group_key=default_group, default_args=default_args)


def register_torch_optimizers():
    for name, module in inspect.getmembers(torch.optim):
        if name.startswith('__'):
            continue
        if inspect.isclass(module) and issubclass(module,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(
                default_group, module_name=name, module_cls=module)


register_torch_optimizers()
