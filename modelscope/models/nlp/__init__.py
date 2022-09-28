# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbones import SbertModel
    from .bart_for_text_error_correction import BartForTextErrorCorrection
    from .bert_for_document_segmentation import BertForDocumentSegmentation
    from .csanmt_for_translation import CsanmtForTranslation
    from .heads import SequenceClassificationHead
    from .gpt3 import GPT3ForTextGeneration
    from .masked_language import (StructBertForMaskedLM, VecoForMaskedLM,
                                  BertForMaskedLM, DebertaV2ForMaskedLM)
    from .ponet_for_masked_language import PoNetForMaskedLM
    from .nncrf_for_named_entity_recognition import (
        TransformerCRFForNamedEntityRecognition,
        LSTMCRFForNamedEntityRecognition)
    from .palm_v2 import PalmForTextGeneration
    from .sbert_for_faq_question_answering import SbertForFaqQuestionAnswering
    from .star_text_to_sql import StarForTextToSql
    from .sequence_classification import (VecoForSequenceClassification,
                                          SbertForSequenceClassification,
                                          BertForSequenceClassification)
    from .space import SpaceForDialogIntent
    from .space import SpaceForDialogModeling
    from .space import SpaceForDialogStateTracking
    from .table_question_answering import TableQuestionAnswering
    from .task_models import (FeatureExtractionModel,
                              InformationExtractionModel,
                              SequenceClassificationModel,
                              SingleBackboneTaskModelBase,
                              TokenClassificationModel)
    from .token_classification import SbertForTokenClassification
    from .sentence_embedding import SentenceEmbedding
    from .passage_ranking import PassageRanking
    from .T5 import T5ForConditionalGeneration
else:
    _import_structure = {
        'backbones': ['SbertModel'],
        'bart_for_text_error_correction': ['BartForTextErrorCorrection'],
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
        'ponet_for_masked_language': ['PoNetForMaskedLM'],
        'palm_v2': ['PalmForTextGeneration'],
        'sbert_for_faq_question_answering': ['SbertForFaqQuestionAnswering'],
        'star_text_to_sql': ['StarForTextToSql'],
        'sequence_classification': [
            'VecoForSequenceClassification', 'SbertForSequenceClassification',
            'BertForSequenceClassification'
        ],
        'space': [
            'SpaceForDialogIntent', 'SpaceForDialogModeling',
            'SpaceForDialogStateTracking'
        ],
        'task_models': [
            'FeatureExtractionModel',
            'InformationExtractionModel',
            'SequenceClassificationModel',
            'SingleBackboneTaskModelBase',
            'TokenClassificationModel',
        ],
        'token_classification': ['SbertForTokenClassification'],
        'table_question_answering': ['TableQuestionAnswering'],
        'sentence_embedding': ['SentenceEmbedding'],
        'passage_ranking': ['PassageRanking'],
        'T5': ['T5ForConditionalGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
