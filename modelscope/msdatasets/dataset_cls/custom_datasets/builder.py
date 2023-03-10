# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg

CUSTOM_DATASETS = Registry('custom_datasets')


def build_custom_dataset(cfg: ConfigDict,
                         task_name: str,
                         default_args: dict = None):
    """ Build custom dataset for user-define dataset given model config and task name.

    Args:
        cfg (:obj:`ConfigDict`): config dict for model object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, CUSTOM_DATASETS, group_key=task_name, default_args=default_args)
