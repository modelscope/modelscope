# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.error import TENSORFLOW_IMPORT_WARNING

try:
    from .translation_pipeline import *  # noqa F403
except ModuleNotFoundError as e:
    if str(e) == "No module named 'tensorflow'":
        print(TENSORFLOW_IMPORT_WARNING.format('translation'))
    else:
        raise ModuleNotFoundError(e)

try:
    from .dialog_intent_prediction_pipeline import *  # noqa F403
    from .dialog_modeling_pipeline import *  # noqa F403
    from .dialog_state_tracking_pipeline import *  # noqa F403
    from .fill_mask_pipeline import *  # noqa F403
    from .named_entity_recognition_pipeline import *  # noqa F403
    from .nli_pipeline import *  # noqa F403
    from .sentence_similarity_pipeline import *  # noqa F403
    from .sentiment_classification_pipeline import *  # noqa F403
    from .sequence_classification_pipeline import *  # noqa F403
    from .text_generation_pipeline import *  # noqa F403
    from .word_segmentation_pipeline import *  # noqa F403
    from .zero_shot_classification_pipeline import *  # noqa F403
except ModuleNotFoundError as e:
    if str(e) == "No module named 'torch'":
        pass
    else:
        raise ModuleNotFoundError(e)
