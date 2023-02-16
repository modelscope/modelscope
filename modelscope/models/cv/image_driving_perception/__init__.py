# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .image_driving_percetion_model import YOLOPv2
    from .preprocessor import ImageDrivingPerceptionPreprocessor
    from .utils import (scale_coords, non_max_suppression,
                        split_for_trace_model, driving_area_mask,
                        lane_line_mask)

else:
    _import_structure = {
        'image_driving_percetion_model': ['YOLOPv2'],
        'preprocessor': ['ImageDrivingPerceptionPreprocessor'],
        'utils': [
            'scale_coords', 'non_max_suppression', 'split_for_trace_model',
            'driving_area_mask', 'lane_line_mask'
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
