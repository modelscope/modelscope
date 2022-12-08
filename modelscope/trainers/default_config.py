# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.config import Config

DEFAULT_CONFIG = {
    'train': {
        'hooks': [
            {
                'type': 'CheckpointHook',
                'interval': 1
            },
            {
                'type': 'TextLoggerHook',
                'interval': 10
            },
            {
                'type': 'IterTimerHook'
            },
            {
                'type': 'TensorboardHook',
                'interval': 10
            },
        ]
    }
}


def merge_cfg(cfg: Config):
    """Merge the default config into the input cfg.

    This function will pop the default CheckpointHook when the BestCkptSaverHook exists in the input cfg.

    Aegs:
        cfg: The input cfg to be merged into.
    """
    cfg.merge_from_dict(DEFAULT_CONFIG, force=False)
    # pop duplicate hook

    if any(['BestCkptSaverHook' == hook['type'] for hook in cfg.train.hooks]):
        cfg.train.hooks = list(
            filter(lambda hook: hook['type'] != 'CheckpointHook',
                   cfg.train.hooks))
