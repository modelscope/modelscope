# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .information_extraction import InformationExtractionModel
    from .sequence_classification import SequenceClassificationModel
    from .task_model import SingleBackboneTaskModelBase

else:
    _import_structure = {
        'information_extraction': ['InformationExtractionModel'],
        'sequence_classification': ['SequenceClassificationModel'],
        'task_model': ['SingleBackboneTaskModelBase'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
