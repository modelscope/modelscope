# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:

    from .models import BaseVideoModel
    from .tada_convnext import TadaConvNeXt
    from .temporal_patch_shift_transformer import PatchShiftTransformer

else:
    _import_structure = {
        'models': ['BaseVideoModel'],
        'tada_convnext': ['TadaConvNeXt'],
        'temporal_patch_shift_transformer': ['PatchShiftTransformer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
