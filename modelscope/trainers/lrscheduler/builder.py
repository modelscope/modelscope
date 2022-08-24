# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg, default_group

LR_SCHEDULER = Registry('lr_scheduler')


def build_lr_scheduler(cfg: ConfigDict, default_args: dict = None):
    """ build lr scheduler from given lr scheduler config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for lr scheduler object.
        default_args (dict, optional): Default initialization arguments.
    """
    if cfg['type'].lower().endswith('warmup'):
        # build warmup lr scheduler
        if not hasattr(cfg, 'base_scheduler'):
            if default_args is None or ('base_scheduler' not in default_args):
                raise ValueError(
                    'Must provide ``base_scheduler`` which is an instance of ``torch.optim.lr_scheduler._LRScheduler`` '
                    'for build warmup lr scheduler.')
    else:
        # build lr scheduler without warmup
        if not hasattr(cfg, 'optimizer'):
            if default_args is None or ('optimizer' not in default_args):
                raise ValueError(
                    'Must provide ``optimizer`` which is an instance of ``torch.optim.Optimizer`` '
                    'for build lr scheduler')

    return build_from_cfg(
        cfg, LR_SCHEDULER, group_key=default_group, default_args=default_args)


def register_torch_lr_scheduler():
    from torch.optim import lr_scheduler
    from torch.optim.lr_scheduler import _LRScheduler

    members = inspect.getmembers(lr_scheduler)

    for name, obj in members:
        if (inspect.isclass(obj) and issubclass(
                obj, _LRScheduler)) or name in ['ReduceLROnPlateau']:
            LR_SCHEDULER.register_module(module_name=name, module_cls=obj)


register_torch_lr_scheduler()
