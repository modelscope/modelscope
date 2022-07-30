# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbones import (SbertModel, SpaceGenerator, SpaceModelBase,
                            GPT3Model)
    from .heads import SequenceClassificationHead
    from .bert_for_sequence_classification import BertForSequenceClassification
    from .csanmt_for_translation import CsanmtForTranslation
    from .masked_language import (StructBertForMaskedLM, VecoForMaskedLM,
                                  BertForMaskedLM)
    from .nncrf_for_named_entity_recognition import TransformerCRFForNamedEntityRecognition
    from .palm_for_text_generation import PalmForTextGeneration
    from .sbert_for_nli import SbertForNLI
    from .sbert_for_sentence_similarity import SbertForSentenceSimilarity
    from .sbert_for_sentiment_classification import SbertForSentimentClassification
    from .sbert_for_token_classification import SbertForTokenClassification
    from .sbert_for_zero_shot_classification import SbertForZeroShotClassification
    from .sequence_classification import SequenceClassificationModel
    from .space_for_dialog_intent_prediction import SpaceForDialogIntent
    from .space_for_dialog_modeling import SpaceForDialogModeling
    from .space_for_dialog_state_tracking import SpaceForDialogStateTracking
    from .task_model import SingleBackboneTaskModelBase
    from .bart_for_text_error_correction import BartForTextErrorCorrection
    from .gpt3_for_text_generation import GPT3ForTextGeneration

else:
    _import_structure = {
        'backbones':
        ['SbertModel', 'SpaceGenerator', 'SpaceModelBase', 'GPT3Model'],
        'heads': ['SequenceClassificationHead'],
        'csanmt_for_translation': ['CsanmtForTranslation'],
        'bert_for_sequence_classification': ['BertForSequenceClassification'],
        'masked_language':
        ['StructBertForMaskedLM', 'VecoForMaskedLM', 'BertForMaskedLM'],
        'nncrf_for_named_entity_recognition':
        ['TransformerCRFForNamedEntityRecognition'],
        'palm_for_text_generation': ['PalmForTextGeneration'],
        'sbert_for_nli': ['SbertForNLI'],
        'sbert_for_sentence_similarity': ['SbertForSentenceSimilarity'],
        'sbert_for_sentiment_classification':
        ['SbertForSentimentClassification'],
        'sbert_for_token_classification': ['SbertForTokenClassification'],
        'sbert_for_zero_shot_classification':
        ['SbertForZeroShotClassification'],
        'sequence_classification': ['SequenceClassificationModel'],
        'space_for_dialog_intent_prediction': ['SpaceForDialogIntent'],
        'space_for_dialog_modeling': ['SpaceForDialogModeling'],
        'space_for_dialog_state_tracking': ['SpaceForDialogStateTracking'],
        'task_model': ['SingleBackboneTaskModelBase'],
        'bart_for_text_error_correction': ['BartForTextErrorCorrection'],
        'gpt3_for_text_generation': ['GPT3ForTextGeneration'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
