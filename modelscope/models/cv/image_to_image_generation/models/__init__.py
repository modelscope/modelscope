# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .autoencoder import VQAutoencoder
    from .clip import VisionTransformer

else:
    _import_structure = {
        'autoencoder': ['VQAutoencoder'],
        'clip': ['VisionTransformer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
