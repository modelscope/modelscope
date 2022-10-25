# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Tasks
from modelscope.utils.registry import TYPE_NAME, Registry, build_from_cfg

MODELS = Registry('models')
BACKBONES = Registry('backbones')
BACKBONES._modules = MODELS._modules
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


def build_backbone(cfg: ConfigDict, default_args: dict = None):
    """ build backbone given backbone config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for backbone object.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, BACKBONES, group_key=Tasks.backbone, default_args=default_args)


def build_head(cfg: ConfigDict,
               task_name: str = None,
               default_args: dict = None):
    """ build head given config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for head object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, HEADS, group_key=task_name, default_args=default_args)
