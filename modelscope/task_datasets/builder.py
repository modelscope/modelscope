# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg

TASK_DATASETS = Registry('task_datasets')


def build_task_dataset(cfg: ConfigDict,
                       task_name: str = None,
                       default_args: dict = None):
    """ Build task specific dataset processor given model config dict and the task name.

    Args:
        cfg (:obj:`ConfigDict`): config dict for model object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, TASK_DATASETS, group_key=task_name, default_args=default_args)
