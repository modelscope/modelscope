# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .image_quality_assessment_degradation_dataset import ImageQualityAssessmentDegradationDataset

else:
    _import_structure = {
        'image_quality_assessment_degradation_dataset':
        ['ImageQualityAssessmentDegradationDataset'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
