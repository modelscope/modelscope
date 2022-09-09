# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .common_utils import SubPreprocessor
    from .parse import get_label
    from .preprocess_dataset import \
        preprocess_dataset
    from .process_dataset import \
        process_dataset, process_tables

else:
    _import_structure = {
        'common_utils': ['SubPreprocessor'],
        'parse': ['get_label'],
        'preprocess_dataset': ['preprocess_dataset'],
        'process_dataset': ['process_dataset', 'process_tables'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
