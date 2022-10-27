# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .detection_pipeline import EasyCVDetectionPipeline
    from .segmentation_pipeline import EasyCVSegmentationPipeline
    from .face_2d_keypoints_pipeline import Face2DKeypointsPipeline
    from .human_wholebody_keypoint_pipeline import HumanWholebodyKeypointsPipeline
else:
    _import_structure = {
        'detection_pipeline': ['EasyCVDetectionPipeline'],
        'segmentation_pipeline': ['EasyCVSegmentationPipeline'],
        'face_2d_keypoints_pipeline': ['Face2DKeypointsPipeline'],
        'human_wholebody_keypoint_pipeline':
        ['HumanWholebodyKeypointsPipeline'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
