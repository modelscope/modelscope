# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbones import SbertModel
    from .heads import SequenceClassificationHead
    from .bert_for_sequence_classification import BertForSequenceClassification
    from .csanmt_for_translation import CsanmtForTranslation
    from .masked_language import (StructBertForMaskedLM, VecoForMaskedLM,
                                  BertForMaskedLM)
    from .nncrf_for_named_entity_recognition import TransformerCRFForNamedEntityRecognition
    from .palm_v2 import PalmForTextGeneration
    from .token_classification import SbertForTokenClassification
    from .sequence_classification import VecoForSequenceClassification, SbertForSequenceClassification
    from .space import SpaceForDialogIntent
    from .space import SpaceForDialogModeling
    from .space import SpaceForDialogStateTracking
    from .task_models.task_model import SingleBackboneTaskModelBase
    from .bart_for_text_error_correction import BartForTextErrorCorrection
    from .gpt3 import GPT3ForTextGeneration
    from .plug import PlugForTextGeneration

else:
    _import_structure = {
        'backbones': ['SbertModel'],
        'heads': ['SequenceClassificationHead'],
        'csanmt_for_translation': ['CsanmtForTranslation'],
        'bert_for_sequence_classification': ['BertForSequenceClassification'],
        'masked_language':
        ['StructBertForMaskedLM', 'VecoForMaskedLM', 'BertForMaskedLM'],
        'nncrf_for_named_entity_recognition':
        ['TransformerCRFForNamedEntityRecognition'],
        'palm_v2': ['PalmForTextGeneration'],
        'token_classification': ['SbertForTokenClassification'],
        'sequence_classification':
        ['VecoForSequenceClassification', 'SbertForSequenceClassification'],
        'space': [
            'SpaceForDialogIntent', 'SpaceForDialogModeling',
            'SpaceForDialogStateTracking'
        ],
        'task_model': ['SingleBackboneTaskModelBase'],
        'bart_for_text_error_correction': ['BartForTextErrorCorrection'],
        'gpt3': ['GPT3ForTextGeneration'],
        'plug': ['PlugForTextGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
