# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .space import SpaceGenerator, SpaceModelBase
    from .structbert import SbertModel
    from .gpt3 import GPT3Model
else:
    _import_structure = {
        'space': ['SpaceGenerator', 'SpaceModelBase'],
        'structbert': ['SbertModel'],
        'gpt3': ['GPT3Model']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
