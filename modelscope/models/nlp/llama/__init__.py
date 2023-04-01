# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import LlamaConfig
    from .text_generation import LlamaForTextGeneration
    from .backbone import LlamaModel
else:
    _import_structure = {
        'configuration': ['LlamaConfig'],
        'text_generation': ['LlamaForTextGeneration'],
        'backbone': ['LlamaModel'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
