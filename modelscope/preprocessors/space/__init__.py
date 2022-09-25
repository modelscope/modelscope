# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .data_loader import DataLoader
    from .dialog_intent_prediction_preprocessor import \
        DialogIntentPredictionPreprocessor
    from .dialog_modeling_preprocessor import DialogModelingPreprocessor
    from .dialog_state_tracking_preprocessor import DialogStateTrackingPreprocessor
    from .dst_processors import InputFeatures
    from .fields import MultiWOZBPETextField, IntentBPETextField

else:
    _import_structure = {
        'data_loader': ['DataLoader'],
        'dialog_intent_prediction_preprocessor':
        ['DialogIntentPredictionPreprocessor'],
        'dialog_modeling_preprocessor': ['DialogModelingPreprocessor'],
        'dialog_state_tracking_preprocessor':
        ['DialogStateTrackingPreprocessor'],
        'dst_processors': ['InputFeatures'],
        'fields': ['MultiWOZBPETextField', 'IntentBPETextField']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
