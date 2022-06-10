# Copyright (c) Alibaba, Inc. and its affiliates.


class Fields(object):
    """ Names for different application fields
    """
    image = 'image'
    video = 'video'
    cv = 'cv'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi-modal'


class Tasks(object):
    """ Names for tasks supported by maas lib.

    Holds the standard task name to use for identifying different tasks.
    This should be used to register models, pipelines, trainers.
    """
    # vision tasks
    image_to_text = 'image-to-text'
    pose_estimation = 'pose-estimation'
    image_classification = 'image-classification'
    image_tagging = 'image-tagging'
    object_detection = 'object-detection'
    image_segmentation = 'image-segmentation'
    image_editing = 'image-editing'
    image_generation = 'image-generation'
    image_matting = 'image-matting'

    # nlp tasks
    sentiment_analysis = 'sentiment-analysis'
    text_classification = 'text-classification'
    relation_extraction = 'relation-extraction'
    zero_shot = 'zero-shot'
    translation = 'translation'
    token_classification = 'token-classification'
    conversational = 'conversational'
    text_generation = 'text-generation'
    dialog_generation = 'dialog-generation'
    dialog_intent = 'dialog-intent'
    table_question_answering = 'table-question-answering'
    feature_extraction = 'feature-extraction'
    sentence_similarity = 'sentence-similarity'
    fill_mask = 'fill-mask '
    summarization = 'summarization'
    question_answering = 'question-answering'

    # audio tasks
    auto_speech_recognition = 'auto-speech-recognition'
    text_to_speech = 'text-to-speech'
    speech_signal_process = 'speech-signal-process'

    # multi-media
    image_captioning = 'image-captioning'
    visual_grounding = 'visual-grounding'
    text_to_image_synthesis = 'text-to-image-synthesis'


class InputFields(object):
    """ Names for input data fileds in the input data for pipelines
    """
    img = 'img'
    text = 'text'
    audio = 'audio'


# configuration filename
# in order to avoid conflict with huggingface
# config file we use maas_config instead
CONFIGFILE = 'maas_config.json'
