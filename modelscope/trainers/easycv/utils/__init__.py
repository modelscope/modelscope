# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .hooks import AddLrLogHook
    from .metric import EasyCVMetric

else:
    _import_structure = {'hooks': ['AddLrLogHook'], 'metric': ['EasyCVMetric']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
