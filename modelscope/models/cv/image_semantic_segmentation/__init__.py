# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .semantic_seg_model import SemanticSegmentation
    from .segformer import Segformer
    from .ddpm_segmentation_model import DDPMSegmentationModel

else:
    _import_structure = {
        'semantic_seg_model': ['SemanticSegmentation'],
        'segformer': ['Segformer'],
        'ddpm_segmentation_model': ['DDPMSegmentationModel']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
