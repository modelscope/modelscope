# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .kws_farfield_dataset import KWSDataset, KWSDataLoader
    from .kws_nearfield_dataset import kws_nearfield_dataset
    from .asr_dataset import ASRDataset

else:
    _import_structure = {
        'kws_farfield_dataset': ['KWSDataset', 'KWSDataLoader'],
        'kws_nearfield_dataset': ['kws_nearfield_dataset'],
        'asr_dataset': ['ASRDataset'],
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
