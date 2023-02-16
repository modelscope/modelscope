# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .swin_transformer import SwinTransformer
    from .swin_transformer import D2SwinTransformer

else:
    _import_structure = {
        'swin_transformer': ['SwinTransformer', 'D2SwinTransformer'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
