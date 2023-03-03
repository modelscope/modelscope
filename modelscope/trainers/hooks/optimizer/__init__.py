# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .apex_optimizer_hook import ApexAMPOptimizerHook
    from .base import OptimizerHook, NoneOptimizerHook
    from .torch_optimizer_hook import TorchAMPOptimizerHook
else:
    _import_structure = {
        'apex_optimizer_hook': ['ApexAMPOptimizerHook'],
        'base': ['OptimizerHook', 'NoneOptimizerHook'],
        'torch_optimizer_hook': ['TorchAMPOptimizerHook']
    }
    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
