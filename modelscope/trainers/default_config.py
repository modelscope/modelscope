# Copyright (c) Alibaba, Inc. and its affiliates.
DEFAULT_CONFIG = {
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
