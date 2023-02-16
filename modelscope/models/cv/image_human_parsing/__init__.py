# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .m2fp_net import M2FP
    from parsing_utils import center_to_target_size_test
else:
    _import_structure = {
        'm2fp_net': ['M2FP'],
        'parsing_utils': ['center_to_target_size_test']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
