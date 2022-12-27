# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .model_resnet_mutex_v4_linewithchar import SegLinkDetector
    from .ops import decode_segments_links_python, combine_segments_python
    from .utils import rboxes_to_polygons, cal_width, nms_python, polygons_from_bitmap
else:
    _import_structure = {
        'model_resnet_mutex_v4_linewithchar': ['SegLinkDetector'],
        'ops': ['decode_segments_links_python', 'combine_segments_python'],
        'utils': [
            'rboxes_to_polygons', 'cal_width', 'nms_python',
            'polygons_from_bitmap'
        ]
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
