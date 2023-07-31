# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .try_on_infer import SALForImageTryOn

else:
    _import_structure = {'try_on_infer': ['SALForImageTryOn']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
