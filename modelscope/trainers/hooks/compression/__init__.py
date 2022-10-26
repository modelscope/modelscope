# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .sparsity_hook import SparsityHook
    from .utils import SparseLinear, convert_sparse_network

else:
    _import_structure = {
        'sparsity_hook': ['SparsityHook'],
        'utils': ['convert_sparse_network', 'SparseLinear'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
