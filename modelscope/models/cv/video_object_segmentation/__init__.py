# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .model import VideoObjectSegmentation

else:
    _import_structure = {'model': ['VideoObjectSegmentation']}

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
