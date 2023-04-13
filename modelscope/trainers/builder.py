# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Trainers
from modelscope.pipelines.builder import normalize_model_input
from modelscope.pipelines.util import is_official_hub_path
from modelscope.utils.config import check_config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.hub import read_config
from modelscope.utils.plugins import (register_modelhub_repo,
                                      register_plugins_repo)
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
    model = default_args.get('model', None)
    model_revision = default_args.get('model_revision', DEFAULT_MODEL_REVISION)

    if isinstance(model, str) \
            or (isinstance(model, list) and isinstance(model[0], str)):
        if is_official_hub_path(model, revision=model_revision):
            # read config file from hub and parse
            configuration = read_config(
                model, revision=model_revision) if isinstance(
                    model, str) else read_config(
                        model[0], revision=model_revision)
            model_dir = normalize_model_input(model, model_revision)
            register_plugins_repo(configuration.safe_get('plugins'))
            register_modelhub_repo(model_dir,
                                   configuration.get('allow_remote', False))
    return build_from_cfg(cfg, TRAINERS, default_args=default_args)
