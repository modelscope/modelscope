# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule, is_torch_available

if TYPE_CHECKING:
    from .base import TaskDataset
    from .builder import TASK_DATASETS, build_task_dataset
    from .torch_base_dataset import TorchTaskDataset

else:
    _import_structure = {
        'base': ['TaskDataset'],
        'builder': ['TASK_DATASETS', 'build_task_dataset'],
        'torch_base_dataset': ['TorchTaskDataset'],
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
