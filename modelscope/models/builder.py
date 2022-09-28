# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import TYPE_NAME, Registry, build_from_cfg

MODELS = Registry('models')
BACKBONES = Registry('backbones')
HEADS = Registry('heads')


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


def build_backbone(cfg: ConfigDict,
                   field: str = None,
                   default_args: dict = None):
    """ build backbone given backbone config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for backbone object.
        field (str, optional): field, such as CV, NLP's backbone
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, BACKBONES, group_key=field, default_args=default_args)


def build_head(cfg: ConfigDict,
               group_key: str = None,
               default_args: dict = None):
    """ build head given config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for head object.
        default_args (dict, optional): Default initialization arguments.
    """
    if group_key is None:
        group_key = cfg[TYPE_NAME]
    return build_from_cfg(
        cfg, HEADS, group_key=group_key, default_args=default_args)
