# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .rrdbnet_arch import RRDBNet
    from .ecbsr_model import ECBSRModel

else:
    _import_structure = {
        'rrdbnet_arch': ['RRDBNet'],
        'ecbsr_model': ['ECBSRModel']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
