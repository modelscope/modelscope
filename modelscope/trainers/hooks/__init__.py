# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .builder import HOOKS, build_hook
    from .checkpoint_hook import BestCkptSaverHook, CheckpointHook, LoadCheckpointHook
    from .early_stop_hook import EarlyStopHook
    from .compression import SparsityHook
    from .evaluation_hook import EvaluationHook
    from .hook import Hook
    from .iter_timer_hook import IterTimerHook
    from .logger import TensorboardHook, TextLoggerHook
    from .lr_scheduler_hook import LrSchedulerHook
    from .optimizer import (ApexAMPOptimizerHook, NoneOptimizerHook,
                            OptimizerHook, TorchAMPOptimizerHook)
    from .priority import Priority, get_priority

else:
    _import_structure = {
        'builder': ['HOOKS', 'build_hook'],
        'checkpoint_hook':
        ['BestCkptSaverHook', 'CheckpointHook', 'LoadCheckpointHook'],
        'compression': ['SparsityHook'],
        'evaluation_hook': ['EvaluationHook'],
        'hook': ['Hook'],
        'iter_timer_hook': ['IterTimerHook'],
        'logger': ['TensorboardHook', 'TextLoggerHook'],
        'lr_scheduler_hook': ['LrSchedulerHook', 'NoneLrSchedulerHook'],
        'optimizer': [
            'ApexAMPOptimizerHook', 'NoneOptimizerHook', 'OptimizerHook',
            'TorchAMPOptimizerHook'
        ],
        'priority': ['Priority', 'get']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
