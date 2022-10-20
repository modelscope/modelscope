# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .tinynas_detector import Tinynas_detector
    from .tinynas_damoyolo import DamoYolo

else:
    _import_structure = {
        'tinynas_detector': ['TinynasDetector'],
        'tinynas_damoyolo': ['DamoYolo'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
