# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .model_zoo import get_zennet

else:
    _import_structure = {
        'model_zoo': ['get_zennet'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
