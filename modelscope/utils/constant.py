# Copyright (c) Alibaba, Inc. and its affiliates.
import enum


class Fields(object):
    """ Names for different application fields
    """
    # image = 'image'
    # video = 'video'
    cv = 'cv'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi-modal'


class Tasks(object):
    """ Names for tasks supported by modelscope.

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
    ocr_detection = 'ocr-detection'
    action_recognition = 'action-recognition'
    video_embedding = 'video-embedding'

    # nlp tasks
    word_segmentation = 'word-segmentation'
    nli = 'nli'
    sentiment_classification = 'sentiment-classification'
    sentiment_analysis = 'sentiment-analysis'
    sentence_similarity = 'sentence-similarity'
    text_classification = 'text-classification'
    relation_extraction = 'relation-extraction'
    zero_shot = 'zero-shot'
    translation = 'translation'
    token_classification = 'token-classification'
    conversational = 'conversational'
    text_generation = 'text-generation'
    dialog_modeling = 'dialog-modeling'
    dialog_intent_prediction = 'dialog-intent-prediction'
    dialog_state_tracking = 'dialog-state-tracking'
    table_question_answering = 'table-question-answering'
    feature_extraction = 'feature-extraction'
    fill_mask = 'fill-mask'
    summarization = 'summarization'
    question_answering = 'question-answering'
    zero_shot_classification = 'zero-shot-classification'

    # audio tasks
    auto_speech_recognition = 'auto-speech-recognition'
    text_to_speech = 'text-to-speech'
    speech_signal_process = 'speech-signal-process'

    # multi-modal tasks
    image_captioning = 'image-captioning'
    visual_grounding = 'visual-grounding'
    text_to_image_synthesis = 'text-to-image-synthesis'
    multi_modal_embedding = 'multi-modal-embedding'
    visual_question_answering = 'visual-question-answering'


class InputFields(object):
    """ Names for input data fields in the input data for pipelines
    """
    img = 'img'
    text = 'text'
    audio = 'audio'


class Hubs(enum.Enum):
    """ Source from which an entity (such as a Dataset or Model) is stored
    """
    modelscope = 'modelscope'
    huggingface = 'huggingface'


class DownloadMode(enum.Enum):
    """ How to treat existing datasets
    """
    REUSE_DATASET_IF_EXISTS = 'reuse_dataset_if_exists'
    FORCE_REDOWNLOAD = 'force_redownload'


class ModelFile(object):
    CONFIGURATION = 'configuration.json'
    README = 'README.md'
    TF_SAVED_MODEL_FILE = 'saved_model.pb'
    TF_GRAPH_FILE = 'tf_graph.pb'
    TF_CHECKPOINT_FOLDER = 'tf_ckpts'
    TF_CKPT_PREFIX = 'ckpt-'
    TORCH_MODEL_FILE = 'pytorch_model.pt'
    TORCH_MODEL_BIN_FILE = 'pytorch_model.bin'


class Requirements(object):
    """Requirement names for each module
    """
    protobuf = 'protobuf'
    sentencepiece = 'sentencepiece'
    sklearn = 'sklearn'
    scipy = 'scipy'
    timm = 'timm'
    tokenizers = 'tokenizers'
    tf = 'tf'
    torch = 'torch'


TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'

DEFAULT_MODEL_REVISION = 'master'
DEFAULT_DATASET_REVISION = 'master'
