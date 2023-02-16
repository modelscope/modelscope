# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Trainers
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Tasks
from modelscope.utils.registry import Registry, build_from_cfg

TRAINERS = Registry('trainers')


def build_trainer(name: str = Trainers.default, default_args: dict = None):
    """ build trainer given a trainer name

    Args:
        name (str, optional):  Trainer name, if None, default trainer
            will be used.
        default_args (dict, optional): Default initialization arguments.
    """
    cfg = dict(type=name)
    return build_from_cfg(cfg, TRAINERS, default_args=default_args)
