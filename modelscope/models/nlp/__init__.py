# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbones import SbertModel
    from .bart_for_text_error_correction import BartForTextErrorCorrection
    from .bert_for_sequence_classification import BertForSequenceClassification
    from .bert_for_document_segmentation import BertForDocumentSegmentation
    from .csanmt_for_translation import CsanmtForTranslation
    from .heads import SequenceClassificationHead
    from .gpt3 import GPT3ForTextGeneration
    from .masked_language import (StructBertForMaskedLM, VecoForMaskedLM,
                                  BertForMaskedLM, DebertaV2ForMaskedLM)
    from .nncrf_for_named_entity_recognition import (
        TransformerCRFForNamedEntityRecognition,
        LSTMCRFForNamedEntityRecognition)
    from .palm_v2 import PalmForTextGeneration
    from .sbert_for_faq_question_answering import SbertForFaqQuestionAnswering
    from .star_text_to_sql import StarForTextToSql
    from .sequence_classification import VecoForSequenceClassification, SbertForSequenceClassification
    from .space import SpaceForDialogIntent
    from .space import SpaceForDialogModeling
    from .space import SpaceForDialogStateTracking
    from .task_models import (InformationExtractionModel,
                              SequenceClassificationModel,
                              SingleBackboneTaskModelBase,
                              TokenClassificationModel)
    from .token_classification import SbertForTokenClassification

else:
    _import_structure = {
        'backbones': ['SbertModel'],
        'bart_for_text_error_correction': ['BartForTextErrorCorrection'],
        'bert_for_sequence_classification': ['BertForSequenceClassification'],
        'bert_for_document_segmentation': ['BertForDocumentSegmentation'],
        'csanmt_for_translation': ['CsanmtForTranslation'],
        'heads': ['SequenceClassificationHead'],
        'gpt3': ['GPT3ForTextGeneration'],
        'masked_language': [
            'StructBertForMaskedLM', 'VecoForMaskedLM', 'BertForMaskedLM',
            'DebertaV2ForMaskedLM'
        ],
        'nncrf_for_named_entity_recognition': [
            'TransformerCRFForNamedEntityRecognition',
            'LSTMCRFForNamedEntityRecognition'
        ],
        'palm_v2': ['PalmForTextGeneration'],
        'sbert_for_faq_question_answering': ['SbertForFaqQuestionAnswering'],
        'star_text_to_sql': ['StarForTextToSql'],
        'sequence_classification':
        ['VecoForSequenceClassification', 'SbertForSequenceClassification'],
        'space': [
            'SpaceForDialogIntent', 'SpaceForDialogModeling',
            'SpaceForDialogStateTracking'
        ],
        'task_models': [
            'InformationExtractionModel', 'SequenceClassificationModel',
            'SingleBackboneTaskModelBase', 'TokenClassificationModel'
        ],
        'token_classification': ['SbertForTokenClassification'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
