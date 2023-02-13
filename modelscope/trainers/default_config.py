# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List, Optional

from modelscope.utils.config import Config

DEFAULT_CONFIG = Config({
    'framework': 'pytorch',
    'train': {
        'work_dir': '/tmp',
        'max_epochs': 10,
        'dataloader': {
            'batch_size_per_gpu': 16,
            'workers_per_gpu': 0
        },
        'optimizer': {
            'type': 'SGD',
            'lr': 1e-3
        },
        'lr_scheduler': {
            'type': 'StepLR',
            'step_size': 2
        },
        'hooks': [{
            'type': 'CheckpointHook',
            'interval': 1
        }]
    },
    'evaluation': {
        'dataloader': {
            'batch_size_per_gpu': 16,
            'workers_per_gpu': 0,
            'shuffle': False
        },
    }
})

DEFAULT_HOOKS_CONFIG = {
    'train': {
        'hooks': [{
            'type': 'CheckpointHook',
            'interval': 1
        }, {
            'type': 'TextLoggerHook',
            'interval': 10
        }, {
            'type': 'IterTimerHook'
        }]
    }
}


def merge_cfg(cfg: Config):
    """Merge the default config into the input cfg.

    This function will pop the default CheckpointHook when the BestCkptSaverHook exists in the input cfg.

    Aegs:
        cfg: The input cfg to be merged into.
    """
    cfg.merge_from_dict(DEFAULT_HOOKS_CONFIG, force=False)


def merge_hooks(cfg: Config) -> List[Dict]:
    key_chain_hook_map = {
        'train.logging': 'TextLoggerHook',
        'train.checkpoint.period': 'CheckpointHook',
        'train.checkpoint.best': 'BestCkptSaverHook',
        'evaluation.period': 'EvaluationHook'
    }
    hooks = cfg.train.hooks.copy()
    for key_chain, hook_type in key_chain_hook_map.items():
        hook = _key_chain_to_hook(cfg, key_chain, hook_type)
        if hook is not None:
            hooks.append(hook)
    return hooks


def _key_chain_to_hook(cfg: Config, key_chain: str,
                       hook_type: str) -> Optional[Dict]:
    if not _check_basic_hook(cfg, key_chain, hook_type):
        return None
    hook_params: Dict = cfg.safe_get(key_chain)
    hook = {'type': hook_type}
    hook.update(hook_params)
    return hook


def _check_basic_hook(cfg: Config, key_chain: str, hook_type: str) -> bool:
    if cfg.safe_get(key_chain) is None:
        return False
    hooks = list(
        filter(lambda hook: hook['type'] == hook_type, cfg.train.hooks))
    assert len(hooks) == 0, f'The key_chain {key_chain} and the traditional hook ' \
                            f'cannot exist at the same time, ' \
                            f'please delete {hook_type} in the configuration file.'
    return True
