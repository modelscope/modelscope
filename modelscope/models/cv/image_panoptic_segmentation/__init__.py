# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .panseg_model import SwinLPanopticSegmentation
    from .r50_panseg_model import R50PanopticSegmentation

else:
    _import_structure = {
        'panseg_model': ['SwinLPanopticSegmentation'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
