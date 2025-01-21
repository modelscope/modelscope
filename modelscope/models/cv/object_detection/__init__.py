# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .mmdet_model import DetectionModel
    from .yolox_pai import YOLOX
    from .dino import DINO

else:
    _import_structure = {
        'mmdet_model': ['DetectionModel'],
        'yolox_pai': ['YOLOX'],
        'dino': ['DINO']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
