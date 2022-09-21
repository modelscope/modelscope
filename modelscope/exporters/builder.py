# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg

EXPORTERS = Registry('exporters')


def build_exporter(cfg: ConfigDict,
                   task_name: str = None,
                   default_args: dict = None):
    """ build exporter by the given model config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for exporter object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, EXPORTERS, group_key=task_name, default_args=default_args)
