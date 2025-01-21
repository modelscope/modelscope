# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .base import Exporter
    from .builder import build_exporter
    from .cv import CartoonTranslationExporter, FaceDetectionSCRFDExporter
    from .multi_modal import StableDiffusionExporter
    from .nlp import (CsanmtForTranslationExporter,
                      SbertForSequenceClassificationExporter,
                      SbertForZeroShotClassificationExporter)
    from .tf_model_exporter import TfModelExporter
    from .torch_model_exporter import TorchModelExporter
else:
    _import_structure = {
        'base': ['Exporter'],
        'builder': ['build_exporter'],
        'cv': ['CartoonTranslationExporter', 'FaceDetectionSCRFDExporter'],
        'multi_modal': ['StableDiffusionExporter'],
        'nlp': [
            'CsanmtForTranslationExporter',
            'SbertForSequenceClassificationExporter',
            'SbertForZeroShotClassificationExporter'
        ],
        'tf_model_exporter': ['TfModelExporter'],
        'torch_model_exporter': ['TorchModelExporter'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
