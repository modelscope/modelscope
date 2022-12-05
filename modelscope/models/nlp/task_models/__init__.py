# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .information_extraction import InformationExtractionModel
    from .feature_extraction import FeatureExtractionModel
    from .fill_mask import FillMaskModel
    from .nncrf_for_named_entity_recognition import (
        LSTMCRFForNamedEntityRecognition,
        LSTMCRFForWordSegmentation,
        LSTMCRFForPartOfSpeech,
        TransformerCRFForNamedEntityRecognition,
        TransformerCRFForWordSegmentation,
    )
    from .sequence_classification import SequenceClassificationModel
    from .task_model import SingleBackboneTaskModelBase
    from .token_classification import TokenClassificationModel
    from .text_generation import TaskModelForTextGeneration

else:
    _import_structure = {
        'information_extraction': ['InformationExtractionModel'],
        'feature_extraction': ['FeatureExtractionModel'],
        'fill_mask': ['FillMaskModel'],
        'nncrf_for_named_entity_recognition': [
            'LSTMCRFForNamedEntityRecognition',
            'LSTMCRFForWordSegmentation',
            'LSTMCRFForPartOfSpeech',
            'TransformerCRFForNamedEntityRecognition',
            'TransformerCRFForWordSegmentation',
        ],
        'sequence_classification': ['SequenceClassificationModel'],
        'task_model': ['SingleBackboneTaskModelBase'],
        'token_classification': ['TokenClassificationModel'],
        'text_generation': ['TaskModelForTextGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
