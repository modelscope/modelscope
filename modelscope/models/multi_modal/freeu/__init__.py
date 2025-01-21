# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
else:
    _import_structure = {
        'free_lunch_utils':
        ['register_free_upblock2d', 'register_free_crossattn_upblock2d']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
