# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .generate_skeleton import gen_skeleton_bvh
    from .utils import (read_obj, write_obj, render, rotate_x, rotate_y,
                        translate, projection)

else:
    _import_structure = {
        'generate_skeleton': ['gen_skeleton_bvh'],
        'utils': [
            'read_obj', 'write_obj', 'render', 'rotate_x', 'rotate_y',
            'translate', 'projection'
        ],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
