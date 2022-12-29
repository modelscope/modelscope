# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional

from megatron_util import initialize_megatron

from modelscope.utils.config import Config
from modelscope.utils.hub import read_config

_DEFAULT_CFG_WITH_MODEL_TYPE = {
    'gpt-moe': {
        'version': 'moe',
        'world_size': 8
    },
    'plug': {
        'version': 'v1',
        'world_size': 8,
        'tensor_model_parallel_size': 8,
        'seed': 1234
    },
    'mglm-text-summarization': {
        'version': 'v1',
        'seed': 1234
    },
}


def init_megatron_util(cfg: Optional[Config] = None,
                       model_dir: Optional[str] = None,
                       **kwargs):
    assert not (cfg is None and model_dir is None), \
        'cfg and model_dir cannot both be None when initializing megatron_util'
    if cfg is None:
        cfg = read_config(model_dir)
    try:
        megatron_cfg = cfg.megatron
    except AttributeError:
        try:
            model_type = cfg.model.type
        except AttributeError:
            # Fit models without model type, such as mglm
            model_type = cfg.pipeline.type
        megatron_cfg = _DEFAULT_CFG_WITH_MODEL_TYPE[model_type] \
            if model_type in _DEFAULT_CFG_WITH_MODEL_TYPE else {}
    megatron_cfg.update(kwargs)
    initialize_megatron(megatron_cfg)
