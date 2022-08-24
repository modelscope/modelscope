# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .convnext import convnext_tiny
    from .vitstr import vitstr_tiny
else:
    _import_structure = {
        'convnext': ['convnext_tiny'],
        'vitstr': ['vitstr_tiny']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
