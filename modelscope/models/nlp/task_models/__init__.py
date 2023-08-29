# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .information_extraction import ModelForInformationExtraction
    from .feature_extraction import ModelForFeatureExtraction
    from .fill_mask import ModelForFillMask
    from .text_classification import ModelForTextClassification
    from .task_model import SingleBackboneTaskModelBase
    from .token_classification import (ModelForTokenClassification,
                                       ModelForTokenClassificationWithCRF)
    from .text_generation import ModelForTextGeneration
    from .text_ranking import ModelForTextRanking
    from .machine_reading_comprehension import ModelForMachineReadingComprehension

else:
    _import_structure = {
        'information_extraction': ['ModelForInformationExtraction'],
        'feature_extraction': ['ModelForFeatureExtraction'],
        'fill_mask': ['ModelForFillMask'],
        'text_classification': ['ModelForTextClassification'],
        'task_model': ['SingleBackboneTaskModelBase'],
        'token_classification':
        ['ModelForTokenClassification', 'ModelForTokenClassificationWithCRF'],
        'text_generation': ['ModelForTextGeneration'],
        'text_ranking': ['ModelForTextRanking'],
        'machine_reading_comprehension':
        ['ModelForMachineReadingComprehension'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
