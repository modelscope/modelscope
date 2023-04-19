# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .video_super_resolution import (VideoReader)
    from .video_stabilization import (stabilization_preprocessor)
    from .mmcls_preprocessor import ImageClassificationMmcvPreprocessor

    from .image_quality_assessment_mos import ImageQualityAssessmentMosPreprocessor
    from .image_quality_assessment_man import ImageQualityAssessmentMANPreprocessor
    from .image_restoration_preprocessor import ImageRestorationPreprocessor
    from .bad_image_detecting_preprocessor import BadImageDetectingPreprocessor
    from .controllable_image_generation import ControllableImageGenerationPreprocessor
    from .image_classification_preprocessor import ImageClassificationPreprocessor

else:
    _import_structure = {
        'video_super_resolution': ['VideoReader'],
        'video_stabilization': ['stabilization_preprocessor'],
        'mmcls_preprocessor': ['ImageClassificationMmcvPreprocessor'],
        'image_quality_assessment_mos':
        ['ImageQualityAssessmentMosPreprocessor'],
        'image_quality_assessment_man':
        ['ImageQualityAssessmentMANPreprocessor'],
        'image_restoration_preprocessor': ['ImageRestorationPreprocessor'],
        'bad_image_detecting_preprocessor': ['BadImageDetectingPreprocessor'],
        'controllable_image_generation':
        ['ControllableImageGenerationPreprocessor'],
        'image_classification_preprocessor':
        ['ImageClassificationPreprocessor'],
        'diffusion_image_generation_preprocessor':
        ['DiffusionImageGenerationPreprocessor']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
