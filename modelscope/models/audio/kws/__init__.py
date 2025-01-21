# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .generic_key_word_spotting import GenericKeyWordSpotting
    from .farfield.model import FSMNSeleNetV2Decorator
    from .nearfield.model import FSMNDecorator

else:
    _import_structure = {
        'generic_key_word_spotting': ['GenericKeyWordSpotting'],
        'farfield.model': ['FSMNSeleNetV2Decorator'],
        'nearfield.model': ['FSMNDecorator'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
