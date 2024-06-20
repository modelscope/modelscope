# Copyright (c) Alibaba, Inc. and its affiliates.
import enum


class Fields(object):
    """ Names for different application fields
    """
    framework = 'framework'
    cv = 'cv'
    nlp = 'nlp'
    audio = 'audio'
    multi_modal = 'multi-modal'
    science = 'science'
    server = 'server'


class CVTasks(object):
    # ocr
    ocr_detection = 'ocr-detection'
    ocr_recognition = 'ocr-recognition'
    table_recognition = 'table-recognition'
    lineless_table_recognition = 'lineless-table-recognition'
    license_plate_detection = 'license-plate-detection'
    card_detection_correction = 'card-detection-correction'

    # human face body related
    animal_recognition = 'animal-recognition'
    face_detection = 'face-detection'
    face_liveness = 'face-liveness'
    face_quality_assessment = 'face-quality-assessment'
    card_detection = 'card-detection'
    face_recognition = 'face-recognition'
    facial_expression_recognition = 'facial-expression-recognition'
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
    pedestrian_attribute_recognition = 'pedestrian-attribute-recognition'

    image_classification = 'image-classification'
    image_multilabel_classification = 'image-multilabel-classification'
    image_classification_imagenet = 'image-classification-imagenet'
    image_classification_dailylife = 'image-classification-dailylife'

    image_object_detection = 'image-object-detection'
    video_object_detection = 'video-object-detection'
    image_fewshot_detection = 'image-fewshot-detection'
    open_vocabulary_detection = 'open-vocabulary-detection'
    object_detection_3d = 'object-detection-3d'

    image_segmentation = 'image-segmentation'
    semantic_segmentation = 'semantic-segmentation'
    image_driving_perception = 'image-driving-perception'
    image_depth_estimation = 'image-depth-estimation'
    dense_optical_flow_estimation = 'dense-optical-flow-estimation'
    image_normal_estimation = 'image-normal-estimation'
    indoor_layout_estimation = 'indoor-layout-estimation'
    video_depth_estimation = 'video-depth-estimation'
    panorama_depth_estimation = 'panorama-depth-estimation'
    portrait_matting = 'portrait-matting'
    universal_matting = 'universal-matting'
    text_driven_segmentation = 'text-driven-segmentation'
    shop_segmentation = 'shop-segmentation'
    hand_static = 'hand-static'
    face_human_hand_detection = 'face-human-hand-detection'
    face_emotion = 'face-emotion'
    product_segmentation = 'product-segmentation'
    image_matching = 'image-matching'
    image_local_feature_matching = 'image-local-feature-matching'
    image_quality_assessment_degradation = 'image-quality-assessment-degradation'

    crowd_counting = 'crowd-counting'

    # image editing
    skin_retouching = 'skin-retouching'
    image_super_resolution = 'image-super-resolution'
    image_super_resolution_pasd = 'image-super-resolution-pasd'
    image_debanding = 'image-debanding'
    image_colorization = 'image-colorization'
    image_color_enhancement = 'image-color-enhancement'
    image_denoising = 'image-denoising'
    image_deblurring = 'image-deblurring'
    image_portrait_enhancement = 'image-portrait-enhancement'
    image_inpainting = 'image-inpainting'
    image_paintbyexample = 'image-paintbyexample'
    image_skychange = 'image-skychange'
    image_demoireing = 'image-demoireing'
    image_editing = 'image-editing'

    # image generation
    image_to_image_translation = 'image-to-image-translation'
    image_to_image_generation = 'image-to-image-generation'
    image_style_transfer = 'image-style-transfer'
    image_portrait_stylization = 'image-portrait-stylization'
    image_body_reshaping = 'image-body-reshaping'
    image_embedding = 'image-embedding'
    image_face_fusion = 'image-face-fusion'
    product_retrieval_embedding = 'product-retrieval-embedding'
    controllable_image_generation = 'controllable-image-generation'
    text_to_360panorama_image = 'text-to-360panorama-image'
    image_try_on = 'image-try-on'
    human_image_generation = 'human-image-generation'
    image_view_transform = 'image-view-transform'

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
    video_panoptic_segmentation = 'video-panoptic-segmentation'
    video_instance_segmentation = 'video-instance-segmentation'

    # video editing
    video_inpainting = 'video-inpainting'
    video_frame_interpolation = 'video-frame-interpolation'
    video_stabilization = 'video-stabilization'
    video_super_resolution = 'video-super-resolution'
    video_deinterlace = 'video-deinterlace'
    video_colorization = 'video-colorization'

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

    # content check
    content_check = 'content-check'

    # 3d face reconstruction
    face_reconstruction = 'face-reconstruction'
    head_reconstruction = 'head-reconstruction'
    text_to_head = 'text-to-head'

    # 3d human reconstruction
    human_reconstruction = 'human-reconstruction'
    text_texture_generation = 'text-texture-generation'

    # image quality assessment mos
    image_quality_assessment_mos = 'image-quality-assessment-mos'
    # motion generation
    motion_generation = 'motion-generation'
    # 3d reconstruction
    nerf_recon_acc = 'nerf-recon-acc'
    nerf_recon_4k = 'nerf-recon-4k'
    nerf_recon_vq_compression = 'nerf-recon-vq-compression'
    surface_recon_common = 'surface-recon-common'
    human3d_render = 'human3d-render'
    human3d_animation = 'human3d-animation'
    image_control_3d_portrait = 'image-control-3d-portrait'
    self_supervised_depth_completion = 'self-supervised-depth-completion'

    # 3d generation
    image_to_3d = 'image-to-3d'

    # vision efficient tuning
    vision_efficient_tuning = 'vision-efficient-tuning'

    # bad image detecting
    bad_image_detecting = 'bad-image-detecting'


class NLPTasks(object):
    # chat
    chat = 'chat'
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
    competency_aware_translation = 'competency-aware-translation'
    token_classification = 'token-classification'
    transformer_crf = 'transformer-crf'
    conversational = 'conversational'
    text_generation = 'text-generation'
    fid_dialogue = 'fid-dialogue'
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
    word_alignment = 'word-alignment'
    faq_question_answering = 'faq-question-answering'
    information_extraction = 'information-extraction'
    document_segmentation = 'document-segmentation'
    extractive_summarization = 'extractive-summarization'
    feature_extraction = 'feature-extraction'
    translation_evaluation = 'translation-evaluation'
    sudoku = 'sudoku'
    text2sql = 'text2sql'
    siamese_uie = 'siamese-uie'
    document_grounded_dialog_retrieval = 'document-grounded-dialog-retrieval'
    document_grounded_dialog_rerank = 'document-grounded-dialog-rerank'
    document_grounded_dialog_generate = 'document-grounded-dialog-generate'
    machine_reading_comprehension = 'machine-reading-comprehension'


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
    speech_language_recognition = 'speech-language-recognition'
    speaker_diarization = 'speaker-diarization'
    audio_quantization = 'audio-quantization'
    voice_activity_detection = 'voice-activity-detection'
    language_score_prediction = 'language-score-prediction'
    speech_timestamp = 'speech-timestamp'
    speaker_diarization_dialogue_detection = 'speaker-diarization-dialogue-detection'
    speaker_diarization_semantic_speaker_turn_detection = 'speaker-diarization-semantic-speaker-turn-detection'
    emotion_recognition = 'emotion-recognition'


class MultiModalTasks(object):
    # multi-modal tasks
    image_captioning = 'image-captioning'
    visual_grounding = 'visual-grounding'
    text_to_image_synthesis = 'text-to-image-synthesis'
    multi_modal_embedding = 'multi-modal-embedding'
    text_video_retrieval = 'text-video-retrieval'
    generative_multi_modal_embedding = 'generative-multi-modal-embedding'
    multi_modal_similarity = 'multi-modal-similarity'
    visual_question_answering = 'visual-question-answering'
    visual_entailment = 'visual-entailment'
    video_multi_modal_embedding = 'video-multi-modal-embedding'
    image_text_retrieval = 'image-text-retrieval'
    document_vl_embedding = 'document-vl-embedding'
    video_captioning = 'video-captioning'
    video_question_answering = 'video-question-answering'
    video_temporal_grounding = 'video-temporal-grounding'
    text_to_video_synthesis = 'text-to-video-synthesis'
    efficient_diffusion_tuning = 'efficient-diffusion-tuning'
    multimodal_dialogue = 'multimodal-dialogue'
    image_to_video = 'image-to-video'
    video_to_video = 'video-to-video'


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
    efficient_diffusion_tuning = 'efficient_diffusion_tuning'


class Tasks(CVTasks, NLPTasks, AudioTasks, MultiModalTasks, ScienceTasks):
    """ Names for tasks supported by modelscope.

    Holds the standard task name to use for identifying different tasks.
    This should be used to register models, pipelines, trainers.
    """
    reverse_field_index = {}
    task_template = 'task-template'

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
    virgo = 'virgo'


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

    # general formation for datasets
    general = 4

    # for local meta cache mark
    formation_mark_ext = '.formation_mark'


DatasetMetaFormats = {
    DatasetFormations.native: ['.json'],
    DatasetFormations.hf_compatible: ['.py'],
    DatasetFormations.general: ['.py'],
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
    TRAIN_BEST_OUTPUT_DIR = 'output_best'
    TS_MODEL_FILE = 'model.ts'
    YAML_FILE = 'model.yaml'
    TOKENIZER_FOLDER = 'tokenizer'
    CONFIG = 'config.json'


class Invoke(object):
    KEY = 'invoked_by'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'


class ThirdParty(object):
    KEY = 'third_party'
    EASYCV = 'easycv'
    ADASEQ = 'adaseq'
    ADADET = 'adadet'


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
    after_init = 'after_init'
    before_run = 'before_run'
    before_val = 'before_val'
    before_train_epoch = 'before_train_epoch'
    before_train_iter = 'before_train_iter'
    after_train_iter = 'after_train_iter'
    after_train_epoch = 'after_train_epoch'
    before_val_epoch = 'before_val_epoch'
    before_val_iter = 'before_val_iter'
    after_val_iter = 'after_val_iter'
    after_val_epoch = 'after_val_epoch'
    after_run = 'after_run'
    after_val = 'after_val'


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

META_FILES_FORMAT = ('.csv', '.jsonl')


class DatasetPathName:
    META_NAME = 'meta'
    DATA_FILES_NAME = 'data_files'
    LOCK_FILE_NAME_ANY = 'any'
    LOCK_FILE_NAME_DELIMITER = '-'


class MetaDataFields:
    ARGS_BIG_DATA = 'big_data'


DatasetVisibilityMap = {1: 'private', 3: 'internal', 5: 'public'}


class DistributedParallelType(object):
    """Parallel Strategies for Distributed Models"""
    DP = 'data_parallel'
    TP = 'tensor_model_parallel'
    PP = 'pipeline_model_parallel'


class DatasetTensorflowConfig:
    BATCH_SIZE = 'batch_size'
    DEFAULT_BATCH_SIZE_VALUE = 5


class VirgoDatasetConfig:

    default_virgo_namespace = 'default_namespace'

    default_dataset_version = '1'

    env_virgo_endpoint = 'VIRGO_ENDPOINT'

    # Columns for meta request
    meta_content = 'metaContent'
    sampling_type = 'samplingType'

    # Columns for meta content
    col_id = 'id'
    col_meta_info = 'meta_info'
    col_analysis_result = 'analysis_result'
    col_external_info = 'external_info'
    col_cache_file = 'cache_file'


DEFAULT_MAXCOMPUTE_ENDPOINT = 'http://service-corp.odps.aliyun-inc.com/api'


class MaxComputeEnvs:

    ACCESS_ID = 'ODPS_ACCESS_ID'

    ACCESS_SECRET_KEY = 'ODPS_ACCESS_SECRET_KEY'

    PROJECT_NAME = 'ODPS_PROJECT_NAME'

    ENDPOINT = 'ODPS_ENDPOINT'
