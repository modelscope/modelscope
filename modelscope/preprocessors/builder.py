# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Fields
from modelscope.utils.registry import Registry, build_from_cfg

PREPROCESSORS = Registry('preprocessors')


def build_preprocessor(cfg: ConfigDict,
                       field_name: str = None,
                       default_args: dict = None):
    """ build preprocessor given model config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for model object.
        field_name (str, optional):  application field name, refer to
            :obj:`Fields` for more details
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, PREPROCESSORS, group_key=field_name, default_args=default_args)
