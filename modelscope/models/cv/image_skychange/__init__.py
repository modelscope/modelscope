# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .skychange_model import ImageSkychange
    from .preprocessor import ImageSkyChangePreprocessor

else:
    _import_structure = {'skychange_model': ['ImageSkychange']}
    _import_structure = {'preprocessor': ['ImageSkyChangePreprocessor']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
