# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .cascade_mask_rcnn_swin import CascadeMaskRCNNSwin
    from .model import CascadeMaskRCNNSwinModel
    from .postprocess_utils import get_img_ins_seg_result
    from .datasets import ImageInstanceSegmentationCocoDataset
else:
    _import_structure = {
        'cascade_mask_rcnn_swin': ['CascadeMaskRCNNSwin'],
        'model': ['CascadeMaskRCNNSwinModel'],
        'postprocess_utils': ['get_img_ins_seg_result'],
        'datasets': ['ImageInstanceSegmentationCocoDataset']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
