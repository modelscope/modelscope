# Copyright (c) Alibaba, Inc. and its affiliates.

from maas_lib.utils.config import ConfigDict
from maas_lib.utils.constant import Tasks
from maas_lib.utils.registry import Registry, build_from_cfg

MODELS = Registry('models')


def build_model(cfg: ConfigDict,
                task_name: str = None,
                default_args: dict = None):
    """ build model given model config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for model object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, MODELS, group_key=task_name, default_args=default_args)
