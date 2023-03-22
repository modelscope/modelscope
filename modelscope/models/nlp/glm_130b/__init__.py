# Modified by Zhipu.AI
# Original Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING, Union

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .text_generation import GLM130bForTextGeneration
else:
    _import_structure = {'text_generation': ['GLM130bForTextGeneration']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
