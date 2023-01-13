# Copyright (c) Alibaba, Inc. and its affiliates.
import enum


class Fields(object):
    """ Names for different application fields
    """
    cv = 'cv'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi-modal'
    science = 'science'


class CVTasks(object):
    # ocr
    ocr_detection = 'ocr-detection'
    ocr_recognition = 'ocr-recognition'
    table_recognition = 'table-recognition'
    license_plate_detection = 'license-plate-detection'

    # human face body related
    animal_recognition = 'animal-recognition'
    face_detection = 'face-detection'
    face_liveness = 'face-liveness'
    card_detection = 'card-detection'
    face_recognition = 'face-recognition'
    face_recognition_ood = 'face-recognition-ood'
    facial_expression_recognition = 'facial-expression-recognition'
    facial_landmark_confidence = 'facial-landmark-confidence'
    face_processing_base = 'face-processing-base'
    face_attribute_recognition = 'face-attribute-recognition'
    face_2d_keypoints = 'face-2d-keypoints'
    human_detection = 'human-detection'
    human_object_interaction = 'human-object-interaction'
    face_image_generation = 'face-image-generation'
    body_2d_keypoints = 'body-2d-keypoints'
    body_3d_keypoints = 'body-3d-keypoints'
    hand_2d_keypoints = 'hand-2d-keypoints'
    general_recognition = 'general-recognition'
    human_wholebody_keypoint = 'human-wholebody-keypoint'

    image_classification = 'image-classification'
    image_multilabel_classification = 'image-multilabel-classification'
    image_classification_imagenet = 'image-classification-imagenet'
    image_classification_dailylife = 'image-classification-dailylife'

    image_object_detection = 'image-object-detection'
    video_object_detection = 'video-object-detection'
    image_fewshot_detection = 'image-fewshot-detection'

    image_segmentation = 'image-segmentation'
    semantic_segmentation = 'semantic-segmentation'
    image_depth_estimation = 'image-depth-estimation'
    indoor_layout_estimation = 'indoor-layout-estimation'
    video_depth_estimation = 'video-depth-estimation'
    panorama_depth_estimation = 'panorama-depth-estimation'
    portrait_matting = 'portrait-matting'
    text_driven_segmentation = 'text-driven-segmentation'
    shop_segmentation = 'shop-segmentation'
    hand_static = 'hand-static'
    face_human_hand_detection = 'face-human-hand-detection'
    face_emotion = 'face-emotion'
    product_segmentation = 'product-segmentation'
    image_matching = 'image-matching'

    crowd_counting = 'crowd-counting'

    # image editing
    skin_retouching = 'skin-retouching'
    image_super_resolution = 'image-super-resolution'
    image_colorization = 'image-colorization'
    image_color_enhancement = 'image-color-enhancement'
    image_denoising = 'image-denoising'
    image_deblurring = 'image-deblurring'
    image_portrait_enhancement = 'image-portrait-enhancement'
    image_inpainting = 'image-inpainting'
    image_skychange = 'image-skychange'

    # image generation
    image_to_image_translation = 'image-to-image-translation'
    image_to_image_generation = 'image-to-image-generation'
    image_style_transfer = 'image-style-transfer'
    image_portrait_stylization = 'image-portrait-stylization'
    image_body_reshaping = 'image-body-reshaping'
    image_embedding = 'image-embedding'
    image_face_fusion = 'image-face-fusion'
    product_retrieval_embedding = 'product-retrieval-embedding'

    # video recognition
    live_category = 'live-category'
    action_recognition = 'action-recognition'
    action_detection = 'action-detection'
    video_category = 'video-category'
    video_embedding = 'video-embedding'
    virtual_try_on = 'virtual-try-on'
    movie_scene_segmentation = 'movie-scene-segmentation'
    language_guided_video_summarization = 'language-guided-video-summarization'
    vop_retrieval = 'video-text-retrieval'

    # video segmentation
    video_object_segmentation = 'video-object-segmentation'
    referring_video_object_segmentation = 'referring-video-object-segmentation'
    video_human_matting = 'video-human-matting'

    # video editing
    video_inpainting = 'video-inpainting'
    video_frame_interpolation = 'video-frame-interpolation'
    video_stabilization = 'video-stabilization'
    video_super_resolution = 'video-super-resolution'

    # reid and tracking
    video_single_object_tracking = 'video-single-object-tracking'
    video_multi_object_tracking = 'video-multi-object-tracking'
    video_summarization = 'video-summarization'
    image_reid_person = 'image-reid-person'

    # pointcloud task
    pointcloud_sceneflow_estimation = 'pointcloud-sceneflow-estimation'
    # image multi-view depth estimation
    image_multi_view_depth_estimation = 'image-multi-view-depth-estimation'

    # domain specific object detection
    domain_specific_object_detection = 'domain-specific-object-detection'


class NLPTasks(object):
    # nlp tasks
    word_segmentation = 'word-segmentation'
    part_of_speech = 'part-of-speech'
    named_entity_recognition = 'named-entity-recognition'
    nli = 'nli'
    sentiment_classification = 'sentiment-classification'
    sentiment_analysis = 'sentiment-analysis'
    sentence_similarity = 'sentence-similarity'
    text_classification = 'text-classification'
    sentence_embedding = 'sentence-embedding'
    text_ranking = 'text-ranking'
    relation_extraction = 'relation-extraction'
    zero_shot = 'zero-shot'
    translation = 'translation'
    token_classification = 'token-classification'
    conversational = 'conversational'
    text_generation = 'text-generation'
    text2text_generation = 'text2text-generation'
    task_oriented_conversation = 'task-oriented-conversation'
    dialog_intent_prediction = 'dialog-intent-prediction'
    dialog_state_tracking = 'dialog-state-tracking'
    table_question_answering = 'table-question-answering'
    fill_mask = 'fill-mask'
    text_summarization = 'text-summarization'
    question_answering = 'question-answering'
    code_translation = 'code-translation'
    code_generation = 'code-generation'
    zero_shot_classification = 'zero-shot-classification'
    backbone = 'backbone'
    text_error_correction = 'text-error-correction'
    faq_question_answering = 'faq-question-answering'
    information_extraction = 'information-extraction'
    document_segmentation = 'document-segmentation'
    extractive_summarization = 'extractive-summarization'
    feature_extraction = 'feature-extraction'
    translation_evaluation = 'translation-evaluation'
    sudoku = 'sudoku'
    text2sql = 'text2sql'


class AudioTasks(object):
    # audio tasks
    auto_speech_recognition = 'auto-speech-recognition'
    text_to_speech = 'text-to-speech'
    speech_signal_process = 'speech-signal-process'
    speech_separation = 'speech-separation'
    acoustic_echo_cancellation = 'acoustic-echo-cancellation'
    acoustic_noise_suppression = 'acoustic-noise-suppression'
    keyword_spotting = 'keyword-spotting'
    inverse_text_processing = 'inverse-text-processing'
    punctuation = 'punctuation'
    speaker_verification = 'speaker-verification'


class MultiModalTasks(object):
    # multi-modal tasks
    image_captioning = 'image-captioning'
    visual_grounding = 'visual-grounding'
    text_to_image_synthesis = 'text-to-image-synthesis'
    multi_modal_embedding = 'multi-modal-embedding'
    generative_multi_modal_embedding = 'generative-multi-modal-embedding'
    multi_modal_similarity = 'multi-modal-similarity'
    visual_question_answering = 'visual-question-answering'
    visual_entailment = 'visual-entailment'
    video_multi_modal_embedding = 'video-multi-modal-embedding'
    image_text_retrieval = 'image-text-retrieval'
    document_vl_embedding = 'document-vl-embedding'
    video_captioning = 'video-captioning'
    video_question_answering = 'video-question-answering'


class ScienceTasks(object):
    protein_structure = 'protein-structure'


class TasksIODescriptions(object):
    image_to_image = 'image_to_image',
    images_to_image = 'images_to_image',
    image_to_text = 'image_to_text',
    seed_to_image = 'seed_to_image',
    text_to_speech = 'text_to_speech',
    text_to_text = 'text_to_text',
    speech_to_text = 'speech_to_text',
    speech_to_speech = 'speech_to_speech'
    speeches_to_speech = 'speeches_to_speech',
    visual_grounding = 'visual_grounding',
    visual_question_answering = 'visual_question_answering',
    visual_entailment = 'visual_entailment',
    generative_multi_modal_embedding = 'generative_multi_modal_embedding'


class Tasks(CVTasks, NLPTasks, AudioTasks, MultiModalTasks, ScienceTasks):
    """ Names for tasks supported by modelscope.

    Holds the standard task name to use for identifying different tasks.
    This should be used to register models, pipelines, trainers.
    """
    reverse_field_index = {}

    @staticmethod
    def find_field_by_task(task_name):
        if len(Tasks.reverse_field_index) == 0:
            # Lazy init, not thread safe
            field_dict = {
                Fields.cv: [
                    getattr(Tasks, attr) for attr in dir(CVTasks)
                    if not attr.startswith('__')
                ],
                Fields.nlp: [
                    getattr(Tasks, attr) for attr in dir(NLPTasks)
                    if not attr.startswith('__')
                ],
                Fields.audio: [
                    getattr(Tasks, attr) for attr in dir(AudioTasks)
                    if not attr.startswith('__')
                ],
                Fields.multi_modal: [
                    getattr(Tasks, attr) for attr in dir(MultiModalTasks)
                    if not attr.startswith('__')
                ],
                Fields.science: [
                    getattr(Tasks, attr) for attr in dir(ScienceTasks)
                    if not attr.startswith('__')
                ],
            }

            for field, tasks in field_dict.items():
                for task in tasks:
                    if task in Tasks.reverse_field_index:
                        raise ValueError(f'Duplicate task: {task}')
                    Tasks.reverse_field_index[task] = field

        return Tasks.reverse_field_index.get(task_name)


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


class DownloadChannel(enum.Enum):
    """ Channels of datasets downloading for uv/pv counting.
    """
    LOCAL = 'local'
    DSW = 'dsw'
    EAIS = 'eais'


class UploadMode(enum.Enum):
    """ How to upload object to remote.
    """
    # Upload all objects from local, existing remote objects may be overwritten. (Default)
    OVERWRITE = 'overwrite'
    # Upload local objects in append mode, skipping all existing remote objects.
    APPEND = 'append'


class DatasetFormations(enum.Enum):
    """ How a dataset is organized and interpreted
    """
    # formation that is compatible with official huggingface dataset, which
    # organizes whole dataset into one single (zip) file.
    hf_compatible = 1
    # native modelscope formation that supports, among other things,
    # multiple files in a dataset
    native = 2
    # for local meta cache mark
    formation_mark_ext = '.formation_mark'


DatasetMetaFormats = {
    DatasetFormations.native: ['.json'],
    DatasetFormations.hf_compatible: ['.py'],
}


class ModelFile(object):
    CONFIGURATION = 'configuration.json'
    README = 'README.md'
    TF_SAVED_MODEL_FILE = 'saved_model.pb'
    TF_GRAPH_FILE = 'tf_graph.pb'
    TF_CHECKPOINT_FOLDER = 'tf_ckpts'
    TF_CKPT_PREFIX = 'ckpt-'
    TORCH_MODEL_FILE = 'pytorch_model.pt'
    TORCH_MODEL_BIN_FILE = 'pytorch_model.bin'
    VOCAB_FILE = 'vocab.txt'
    ONNX_MODEL_FILE = 'model.onnx'
    LABEL_MAPPING = 'label_mapping.json'
    TRAIN_OUTPUT_DIR = 'output'
    TS_MODEL_FILE = 'model.ts'
    YAML_FILE = 'model.yaml'
    TOKENIZER_FOLDER = 'tokenizer'


class Invoke(object):
    KEY = 'invoked_by'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'


class ConfigFields(object):
    """ First level keyword in configuration file
    """
    framework = 'framework'
    task = 'task'
    pipeline = 'pipeline'
    model = 'model'
    dataset = 'dataset'
    preprocessor = 'preprocessor'
    train = 'train'
    evaluation = 'evaluation'
    postprocessor = 'postprocessor'


class ConfigKeys(object):
    """Fixed keywords in configuration file"""
    train = 'train'
    val = 'val'
    test = 'test'


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


class Frameworks(object):
    tf = 'tensorflow'
    torch = 'pytorch'
    kaldi = 'kaldi'


DEFAULT_MODEL_REVISION = None
MASTER_MODEL_BRANCH = 'master'
DEFAULT_REPOSITORY_REVISION = 'master'
DEFAULT_DATASET_REVISION = 'master'
DEFAULT_DATASET_NAMESPACE = 'modelscope'
DEFAULT_DATA_ACCELERATION_ENDPOINT = 'https://oss-accelerate.aliyuncs.com'


class ModeKeys:
    TRAIN = 'train'
    EVAL = 'eval'
    INFERENCE = 'inference'


class LogKeys:
    ITER = 'iter'
    ITER_TIME = 'iter_time'
    EPOCH = 'epoch'
    LR = 'lr'  # learning rate
    MODE = 'mode'
    DATA_LOAD_TIME = 'data_load_time'
    ETA = 'eta'  # estimated time of arrival
    MEMORY = 'memory'
    LOSS = 'loss'


class TrainerStages:
    before_run = 'before_run'
    before_train_epoch = 'before_train_epoch'
    before_train_iter = 'before_train_iter'
    after_train_iter = 'after_train_iter'
    after_train_epoch = 'after_train_epoch'
    before_val_epoch = 'before_val_epoch'
    before_val_iter = 'before_val_iter'
    after_val_iter = 'after_val_iter'
    after_val_epoch = 'after_val_epoch'
    after_run = 'after_run'


class ColorCodes:
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'


class Devices:
    """device used for training and inference"""
    cpu = 'cpu'
    gpu = 'gpu'


# Supported extensions for text datasets.
EXTENSIONS_TO_LOAD = {
    'csv': 'csv',
    'tsv': 'csv',
    'json': 'json',
    'jsonl': 'json',
    'parquet': 'parquet',
    'txt': 'text'
}


class DatasetPathName:
    META_NAME = 'meta'
    DATA_FILES_NAME = 'data_files'
    LOCK_FILE_NAME_ANY = 'any'
    LOCK_FILE_NAME_DELIMITER = '-'


class MetaDataFields:
    ARGS_BIG_DATA = 'big_data'


DatasetVisibilityMap = {1: 'private', 3: 'internal', 5: 'public'}
