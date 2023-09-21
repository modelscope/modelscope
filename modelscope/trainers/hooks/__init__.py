# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .builder import HOOKS, build_hook
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
    from .checkpoint import CheckpointHook, LoadCheckpointHook, BestCkptSaverHook
    from .distributed.ddp_hook import DDPHook
    from .distributed.deepspeed_hook import DeepspeedHook
    from .distributed.megatron_hook import MegatronHook
    from .swift.swift_hook import SwiftHook

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
        'checkpoint':
        ['CheckpointHook', 'LoadCheckpointHook', 'BestCkptSaverHook'],
        'distributed.ddp_hook': ['DDPHook'],
        'distributed.deepspeed_hook': ['DeepspeedHook'],
        'distributed.megatron_hook': ['MegatronHook'],
        'swift.swift_hook': ['SwiftHook'],
        'priority': ['Priority', 'get_priority']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
