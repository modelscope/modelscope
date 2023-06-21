# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List, Optional, Tuple

from modelscope.utils.config import Config

DEFAULT_HOOKS_CONFIG = {
    'train.hooks': [{
        'type': 'CheckpointHook',
        'interval': 1
    }, {
        'type': 'TextLoggerHook',
        'interval': 10
    }, {
        'type': 'IterTimerHook'
    }]
}

_HOOK_KEY_CHAIN_MAP = {
    'TextLoggerHook': 'train.logging',
    'CheckpointHook': 'train.checkpoint.period',
    'BestCkptSaverHook': 'train.checkpoint.best',
    'EvaluationHook': 'evaluation.period',
}


def merge_cfg(cfg: Config):
    """Merge the default config into the input cfg.

    This function will pop the default CheckpointHook when the BestCkptSaverHook exists in the input cfg.

    Aegs:
        cfg: The input cfg to be merged into.
    """
    cfg.merge_from_dict(DEFAULT_HOOKS_CONFIG, force=False)


def merge_hooks(cfg: Config) -> List[Dict]:
    hooks = getattr(cfg.train, 'hooks', []).copy()
    for hook_type, key_chain in _HOOK_KEY_CHAIN_MAP.items():
        hook = _key_chain_to_hook(cfg, key_chain, hook_type)
        if hook is not None:
            hooks.append(hook)
    return hooks


def update_cfg(cfg: Config) -> Config:
    if 'hooks' not in cfg.train:
        return cfg
    key_chain_map = {}
    for hook in cfg.train.hooks:
        if not hook:
            continue
        key, value = _hook_split(hook)
        if key not in _HOOK_KEY_CHAIN_MAP:
            continue
        key_chain_map[_HOOK_KEY_CHAIN_MAP[key]] = value
        hook.clear()
    cfg.train.hooks = list(filter(bool, cfg.train.hooks))
    cfg.merge_from_dict(key_chain_map, force=False)
    return cfg


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
        filter(lambda hook: hook['type'] == hook_type,
               getattr(cfg.train, 'hooks', [])))
    assert len(hooks) == 0, f'The key_chain {key_chain} and the traditional hook ' \
                            f'cannot exist at the same time, ' \
                            f'please delete {hook_type} in the configuration file.'
    return True


def _hook_split(hook: Dict) -> Tuple[str, Dict]:
    hook = hook.copy()
    return hook.pop('type'), hook
