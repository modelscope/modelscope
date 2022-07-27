# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.utils.error import TENSORFLOW_IMPORT_WARNING
from .backbones import *  # noqa F403
from .bert_for_sequence_classification import *  # noqa F403
from .heads import *  # noqa F403
from .masked_language import *  # noqa F403
from .nncrf_for_named_entity_recognition import *  # noqa F403
from .palm_for_text_generation import *  # noqa F403
from .sbert_for_nli import *  # noqa F403
from .sbert_for_sentence_similarity import *  # noqa F403
from .sbert_for_sentiment_classification import *  # noqa F403
from .sbert_for_token_classification import *  # noqa F403
from .sbert_for_zero_shot_classification import *  # noqa F403
from .sequence_classification import *  # noqa F403
from .space_for_dialog_intent_prediction import *  # noqa F403
from .space_for_dialog_modeling import *  # noqa F403
from .space_for_dialog_state_tracking import *  # noqa F403

try:
    from .csanmt_for_translation import CsanmtForTranslation
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        print(TENSORFLOW_IMPORT_WARNING.format('CsanmtForTranslation'))
    else:
        raise ModuleNotFoundError(e)
