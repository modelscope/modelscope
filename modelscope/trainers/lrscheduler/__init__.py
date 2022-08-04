# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .builder import LR_SCHEDULER, build_lr_scheduler
    from .warmup import BaseWarmup, ConstantWarmup, ExponentialWarmup, LinearWarmup

else:
    _import_structure = {
        'builder': ['LR_SCHEDULER', 'build_lr_scheduler'],
        'warmup':
        ['BaseWarmup', 'ConstantWarmup', 'ExponentialWarmup', 'LinearWarmup']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
