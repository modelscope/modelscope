try:
    from .dialog_intent_prediction_pipeline import *  # noqa F403
    from .dialog_modeling_pipeline import *  # noqa F403
    from .dialog_state_tracking_pipeline import *  # noqa F403
    from .fill_mask_pipeline import *  # noqa F403
    from .nli_pipeline import *  # noqa F403
    from .sentence_similarity_pipeline import *  # noqa F403
    from .sentiment_classification_pipeline import *  # noqa F403
    from .sequence_classification_pipeline import *  # noqa F403
    from .text_generation_pipeline import *  # noqa F403
    from .translation_pipeline import *  # noqa F403
    from .word_segmentation_pipeline import *  # noqa F403
    from .zero_shot_classification_pipeline import *  # noqa F403
except ModuleNotFoundError as e:
    if str(e) == "No module named 'torch'":
        pass
    else:
        raise ModuleNotFoundError(e)
