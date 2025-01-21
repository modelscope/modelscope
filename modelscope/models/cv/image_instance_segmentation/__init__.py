# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .cascade_mask_rcnn_swin import CascadeMaskRCNNSwin
    from .maskdino_swin import MaskDINOSwin
    from .model import CascadeMaskRCNNSwinModel
    from .maskdino_model import MaskDINOSwinModel
    from .fastinst_model import FastInst
    from .postprocess_utils import get_img_ins_seg_result, get_maskdino_ins_seg_result
else:
    _import_structure = {
        'cascade_mask_rcnn_swin': ['CascadeMaskRCNNSwin'],
        'fastinst_model': ['FastInst'],
        'maskdino_swin': ['MaskDINOSwin'],
        'model': ['CascadeMaskRCNNSwinModel'],
        'maskdino_model': ['MaskDINOSwinModel'],
        'postprocess_utils':
        ['get_img_ins_seg_result', 'get_maskdino_ins_seg_result'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
