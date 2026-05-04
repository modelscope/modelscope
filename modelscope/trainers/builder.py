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
            If ``trust_remote_code`` key is set to True in default_args,
            remote code and plugins declared in the model configuration
            will be allowed to execute.
    """
    cfg = dict(type=name)
    default_args = default_args or {}
    model = default_args.get('model', None)
    model_revision = default_args.get('model_revision', DEFAULT_MODEL_REVISION)
    trust_remote_code = default_args.get('trust_remote_code', False)

    if isinstance(model, str) \
            or (isinstance(model, list) and isinstance(model[0], str)):
        if is_official_hub_path(model, revision=model_revision):
            # read config file from hub and parse
            configuration = read_config(
                model, revision=model_revision) if isinstance(
                    model, str) else read_config(
                        model[0], revision=model_revision)
            model_dir = normalize_model_input(model, model_revision)
            if configuration:
                plugins = configuration.safe_get('plugins')
                allow_remote = configuration.get('allow_remote', False)
                if (plugins or allow_remote) and not trust_remote_code:
                    raise RuntimeError(
                        'Detected plugins or allow_remote field in the model '
                        'configuration file, but trust_remote_code=True was '
                        'not explicitly set.\n'
                        'To prevent potential execution of malicious code, '
                        'loading has been refused.\n'
                        'If you trust this model repository, please pass '
                        'trust_remote_code=True in default_args to '
                        'build_trainer().')
                register_plugins_repo(plugins)
                model_dir_str = model_dir if isinstance(model_dir,
                                                        str) else model_dir[0]
                register_modelhub_repo(model_dir_str, trust_remote_code
                                       and allow_remote)
    return build_from_cfg(cfg, TRAINERS, default_args=default_args)
