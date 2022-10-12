# Copyright (c) Alibaba, Inc. and its affiliates.


class Models(object):
    """ Names for different models.

        Holds the standard model name to use for identifying different model.
    This should be used to register models.

        Model name should only contain model info but not task info.
    """
    tinynas_detection = 'tinynas-detection'

    # vision models
    detection = 'detection'
    realtime_object_detection = 'realtime-object-detection'
    scrfd = 'scrfd'
    classification_model = 'ClassificationModel'
    nafnet = 'nafnet'
    csrnet = 'csrnet'
    cascade_mask_rcnn_swin = 'cascade_mask_rcnn_swin'
    gpen = 'gpen'
    product_retrieval_embedding = 'product-retrieval-embedding'
    body_2d_keypoints = 'body-2d-keypoints'
    body_3d_keypoints = 'body-3d-keypoints'
    crowd_counting = 'HRNetCrowdCounting'
    face_2d_keypoints = 'face-2d-keypoints'
    panoptic_segmentation = 'swinL-panoptic-segmentation'
    image_reid_person = 'passvitb'
    image_inpainting = 'FFTInpainting'
    video_summarization = 'pgl-video-summarization'
    swinL_semantic_segmentation = 'swinL-semantic-segmentation'
    vitadapter_semantic_segmentation = 'vitadapter-semantic-segmentation'
    text_driven_segmentation = 'text-driven-segmentation'
    resnet50_bert = 'resnet50-bert'
    fer = 'fer'
    retinaface = 'retinaface'
    shop_segmentation = 'shop-segmentation'
    mogface = 'mogface'
    mtcnn = 'mtcnn'
    ulfd = 'ulfd'
    video_inpainting = 'video-inpainting'
    human_wholebody_keypoint = 'human-wholebody-keypoint'
    hand_static = 'hand-static'
    face_human_hand_detection = 'face-human-hand-detection'
    face_emotion = 'face-emotion'
    product_segmentation = 'product-segmentation'
    image_body_reshaping = 'image-body-reshaping'

    # EasyCV models
    yolox = 'YOLOX'
    segformer = 'Segformer'
    image_object_detection_auto = 'image-object-detection-auto'

    # nlp models
    bert = 'bert'
    palm = 'palm-v2'
    structbert = 'structbert'
    deberta_v2 = 'deberta_v2'
    veco = 'veco'
    translation = 'csanmt-translation'
    space_dst = 'space-dst'
    space_intent = 'space-intent'
    space_modeling = 'space-modeling'
    star = 'star'
    star3 = 'star3'
    tcrf = 'transformer-crf'
    transformer_softmax = 'transformer-softmax'
    lcrf = 'lstm-crf'
    gcnncrf = 'gcnn-crf'
    bart = 'bart'
    gpt3 = 'gpt3'
    plug = 'plug'
    bert_for_ds = 'bert-for-document-segmentation'
    ponet = 'ponet'
    T5 = 'T5'

    # audio models
    sambert_hifigan = 'sambert-hifigan'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    kws_kwsbp = 'kws-kwsbp'
    generic_asr = 'generic-asr'

    # multi-modal models
    ofa = 'ofa'
    clip = 'clip-multi-modal-embedding'
    gemm = 'gemm-generative-multi-modal'
    mplug = 'mplug'
    diffusion = 'diffusion-text-to-image-synthesis'
    multi_stage_diffusion = 'multi-stage-diffusion-text-to-image-synthesis'
    team = 'team-multi-modal-similarity'
    video_clip = 'video-clip-multi-modal-embedding'


class TaskModels(object):
    # nlp task
    text_classification = 'text-classification'
    token_classification = 'token-classification'
    information_extraction = 'information-extraction'
    fill_mask = 'fill-mask'
    feature_extraction = 'feature-extraction'


class Heads(object):
    # nlp heads

    # text cls
    text_classification = 'text-classification'
    # fill mask
    fill_mask = 'fill-mask'
    bert_mlm = 'bert-mlm'
    roberta_mlm = 'roberta-mlm'
    # token cls
    token_classification = 'token-classification'
    # extraction
    information_extraction = 'information-extraction'


class Pipelines(object):
    """ Names for different pipelines.

        Holds the standard pipline name to use for identifying different pipeline.
    This should be used to register pipelines.

        For pipeline which support different models and implements the common function, we
    should use task name for this pipeline.
        For pipeline which suuport only one model, we should use ${Model}-${Task} as its name.
    """
    # vision tasks
    portrait_matting = 'unet-image-matting'
    image_denoise = 'nafnet-image-denoise'
    person_image_cartoon = 'unet-person-image-cartoon'
    ocr_detection = 'resnet18-ocr-detection'
    action_recognition = 'TAdaConv_action-recognition'
    animal_recognition = 'resnet101-animal-recognition'
    general_recognition = 'resnet101-general-recognition'
    cmdssl_video_embedding = 'cmdssl-r2p1d_video_embedding'
    hicossl_video_embedding = 'hicossl-s3dg-video_embedding'
    body_2d_keypoints = 'hrnetv2w32_body-2d-keypoints_image'
    body_3d_keypoints = 'canonical_body-3d-keypoints_video'
    hand_2d_keypoints = 'hrnetv2w18_hand-2d-keypoints_image'
    human_detection = 'resnet18-human-detection'
    object_detection = 'vit-object-detection'
    easycv_detection = 'easycv-detection'
    easycv_segmentation = 'easycv-segmentation'
    face_2d_keypoints = 'mobilenet_face-2d-keypoints_alignment'
    salient_detection = 'u2net-salient-detection'
    image_classification = 'image-classification'
    face_detection = 'resnet-face-detection-scrfd10gkps'
    ulfd_face_detection = 'manual-face-detection-ulfd'
    facial_expression_recognition = 'vgg19-facial-expression-recognition-fer'
    retina_face_detection = 'resnet50-face-detection-retinaface'
    mog_face_detection = 'resnet101-face-detection-cvpr22papermogface'
    mtcnn_face_detection = 'manual-face-detection-mtcnn'
    live_category = 'live-category'
    general_image_classification = 'vit-base_image-classification_ImageNet-labels'
    daily_image_classification = 'vit-base_image-classification_Dailylife-labels'
    image_color_enhance = 'csrnet-image-color-enhance'
    virtual_try_on = 'virtual-try-on'
    image_colorization = 'unet-image-colorization'
    image_style_transfer = 'AAMS-style-transfer'
    image_super_resolution = 'rrdb-image-super-resolution'
    face_image_generation = 'gan-face-image-generation'
    product_retrieval_embedding = 'resnet50-product-retrieval-embedding'
    realtime_object_detection = 'cspnet_realtime-object-detection_yolox'
    face_recognition = 'ir101-face-recognition-cfglint'
    image_instance_segmentation = 'cascade-mask-rcnn-swin-image-instance-segmentation'
    image2image_translation = 'image-to-image-translation'
    live_category = 'live-category'
    video_category = 'video-category'
    ocr_recognition = 'convnextTiny-ocr-recognition'
    image_portrait_enhancement = 'gpen-image-portrait-enhancement'
    image_to_image_generation = 'image-to-image-generation'
    image_object_detection_auto = 'yolox_image-object-detection-auto'
    skin_retouching = 'unet-skin-retouching'
    tinynas_classification = 'tinynas-classification'
    tinynas_detection = 'tinynas-detection'
    crowd_counting = 'hrnet-crowd-counting'
    action_detection = 'ResNetC3D-action-detection'
    video_single_object_tracking = 'ostrack-vitb-video-single-object-tracking'
    image_panoptic_segmentation = 'image-panoptic-segmentation'
    video_summarization = 'googlenet_pgl_video_summarization'
    image_semantic_segmentation = 'image-semantic-segmentation'
    image_reid_person = 'passvitb-image-reid-person'
    image_inpainting = 'fft-inpainting'
    text_driven_segmentation = 'text-driven-segmentation'
    movie_scene_segmentation = 'resnet50-bert-movie-scene-segmentation'
    shop_segmentation = 'shop-segmentation'
    video_inpainting = 'video-inpainting'
    human_wholebody_keypoint = 'hrnetw48_human-wholebody-keypoint_image'
    pst_action_recognition = 'patchshift-action-recognition'
    hand_static = 'hand-static'
    face_human_hand_detection = 'face-human-hand-detection'
    face_emotion = 'face-emotion'
    product_segmentation = 'product-segmentation'
    image_body_reshaping = 'flow-based-body-reshaping'

    # nlp tasks
    automatic_post_editing = 'automatic-post-editing'
    translation_quality_estimation = 'translation-quality-estimation'
    domain_classification = 'domain-classification'
    sentence_similarity = 'sentence-similarity'
    word_segmentation = 'word-segmentation'
    part_of_speech = 'part-of-speech'
    named_entity_recognition = 'named-entity-recognition'
    text_generation = 'text-generation'
    text2text_generation = 'text2text-generation'
    sentiment_analysis = 'sentiment-analysis'
    sentiment_classification = 'sentiment-classification'
    text_classification = 'text-classification'
    fill_mask = 'fill-mask'
    fill_mask_ponet = 'fill-mask-ponet'
    csanmt_translation = 'csanmt-translation'
    nli = 'nli'
    dialog_intent_prediction = 'dialog-intent-prediction'
    dialog_modeling = 'dialog-modeling'
    dialog_state_tracking = 'dialog-state-tracking'
    zero_shot_classification = 'zero-shot-classification'
    text_error_correction = 'text-error-correction'
    plug_generation = 'plug-generation'
    faq_question_answering = 'faq-question-answering'
    conversational_text_to_sql = 'conversational-text-to-sql'
    table_question_answering_pipeline = 'table-question-answering-pipeline'
    sentence_embedding = 'sentence-embedding'
    passage_ranking = 'passage-ranking'
    relation_extraction = 'relation-extraction'
    document_segmentation = 'document-segmentation'
    feature_extraction = 'feature-extraction'

    # audio tasks
    sambert_hifigan_tts = 'sambert-hifigan-tts'
    speech_dfsmn_aec_psm_16k = 'speech-dfsmn-aec-psm-16k'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    kws_kwsbp = 'kws-kwsbp'
    asr_inference = 'asr-inference'

    # multi-modal tasks
    image_captioning = 'image-captioning'
    multi_modal_embedding = 'multi-modal-embedding'
    generative_multi_modal_embedding = 'generative-multi-modal-embedding'
    visual_question_answering = 'visual-question-answering'
    visual_grounding = 'visual-grounding'
    visual_entailment = 'visual-entailment'
    multi_modal_similarity = 'multi-modal-similarity'
    text_to_image_synthesis = 'text-to-image-synthesis'
    video_multi_modal_embedding = 'video-multi-modal-embedding'
    image_text_retrieval = 'image-text-retrieval'


class Trainers(object):
    """ Names for different trainer.

        Holds the standard trainer name to use for identifying different trainer.
    This should be used to register trainers.

        For a general Trainer, you can use EpochBasedTrainer.
        For a model specific Trainer, you can use ${ModelName}-${Task}-trainer.
    """

    default = 'trainer'
    easycv = 'easycv'

    # multi-modal trainers
    clip_multi_modal_embedding = 'clip-multi-modal-embedding'

    # cv trainers
    image_instance_segmentation = 'image-instance-segmentation'
    image_portrait_enhancement = 'image-portrait-enhancement'
    video_summarization = 'video-summarization'
    movie_scene_segmentation = 'movie-scene-segmentation'
    image_inpainting = 'image-inpainting'

    # nlp trainers
    bert_sentiment_analysis = 'bert-sentiment-analysis'
    dialog_modeling_trainer = 'dialog-modeling-trainer'
    dialog_intent_trainer = 'dialog-intent-trainer'
    nlp_base_trainer = 'nlp-base-trainer'
    nlp_veco_trainer = 'nlp-veco-trainer'
    nlp_passage_ranking_trainer = 'nlp-passage-ranking-trainer'

    # audio trainers
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'


class Preprocessors(object):
    """ Names for different preprocessor.

        Holds the standard preprocessor name to use for identifying different preprocessor.
    This should be used to register preprocessors.

        For a general preprocessor, just use the function name as preprocessor name such as
    resize-image, random-crop
        For a model-specific preprocessor, use ${modelname}-${fuction}
    """

    # cv preprocessor
    load_image = 'load-image'
    image_denoie_preprocessor = 'image-denoise-preprocessor'
    image_color_enhance_preprocessor = 'image-color-enhance-preprocessor'
    image_instance_segmentation_preprocessor = 'image-instance-segmentation-preprocessor'
    image_portrait_enhancement_preprocessor = 'image-portrait-enhancement-preprocessor'
    video_summarization_preprocessor = 'video-summarization-preprocessor'
    movie_scene_segmentation_preprocessor = 'movie-scene-segmentation-preprocessor'

    # nlp preprocessor
    sen_sim_tokenizer = 'sen-sim-tokenizer'
    cross_encoder_tokenizer = 'cross-encoder-tokenizer'
    bert_seq_cls_tokenizer = 'bert-seq-cls-tokenizer'
    text_gen_tokenizer = 'text-gen-tokenizer'
    text2text_gen_preprocessor = 'text2text-gen-preprocessor'
    token_cls_tokenizer = 'token-cls-tokenizer'
    ner_tokenizer = 'ner-tokenizer'
    nli_tokenizer = 'nli-tokenizer'
    sen_cls_tokenizer = 'sen-cls-tokenizer'
    dialog_intent_preprocessor = 'dialog-intent-preprocessor'
    dialog_modeling_preprocessor = 'dialog-modeling-preprocessor'
    dialog_state_tracking_preprocessor = 'dialog-state-tracking-preprocessor'
    sbert_token_cls_tokenizer = 'sbert-token-cls-tokenizer'
    zero_shot_cls_tokenizer = 'zero-shot-cls-tokenizer'
    text_error_correction = 'text-error-correction'
    sentence_embedding = 'sentence-embedding'
    passage_ranking = 'passage-ranking'
    sequence_labeling_tokenizer = 'sequence-labeling-tokenizer'
    word_segment_text_to_label_preprocessor = 'word-segment-text-to-label-preprocessor'
    fill_mask = 'fill-mask'
    fill_mask_ponet = 'fill-mask-ponet'
    faq_question_answering_preprocessor = 'faq-question-answering-preprocessor'
    conversational_text_to_sql = 'conversational-text-to-sql'
    table_question_answering_preprocessor = 'table-question-answering-preprocessor'
    re_tokenizer = 're-tokenizer'
    document_segmentation = 'document-segmentation'
    feature_extraction = 'feature-extraction'

    # audio preprocessor
    linear_aec_fbank = 'linear-aec-fbank'
    text_to_tacotron_symbols = 'text-to-tacotron-symbols'
    wav_to_lists = 'wav-to-lists'
    wav_to_scp = 'wav-to-scp'

    # multi-modal preprocessor
    ofa_tasks_preprocessor = 'ofa-tasks-preprocessor'
    mplug_tasks_preprocessor = 'mplug-tasks-preprocessor'


class Metrics(object):
    """ Names for different metrics.
    """

    # accuracy
    accuracy = 'accuracy'
    audio_noise_metric = 'audio-noise-metric'

    # metrics for image denoise task
    image_denoise_metric = 'image-denoise-metric'

    # metric for image instance segmentation task
    image_ins_seg_coco_metric = 'image-ins-seg-coco-metric'
    # metrics for sequence classification task
    seq_cls_metric = 'seq-cls-metric'
    # metrics for token-classification task
    token_cls_metric = 'token-cls-metric'
    # metrics for text-generation task
    text_gen_metric = 'text-gen-metric'
    # metrics for image-color-enhance task
    image_color_enhance_metric = 'image-color-enhance-metric'
    # metrics for image-portrait-enhancement task
    image_portrait_enhancement_metric = 'image-portrait-enhancement-metric'
    video_summarization_metric = 'video-summarization-metric'
    # metric for movie-scene-segmentation task
    movie_scene_segmentation_metric = 'movie-scene-segmentation-metric'
    # metric for inpainting task
    image_inpainting_metric = 'image-inpainting-metric'


class Optimizers(object):
    """ Names for different OPTIMIZER.

        Holds the standard optimizer name to use for identifying different optimizer.
        This should be used to register optimizer.
    """

    default = 'optimizer'

    SGD = 'SGD'


class Hooks(object):
    """ Names for different hooks.

        All kinds of hooks are defined here
    """
    # lr
    LrSchedulerHook = 'LrSchedulerHook'
    PlateauLrSchedulerHook = 'PlateauLrSchedulerHook'
    NoneLrSchedulerHook = 'NoneLrSchedulerHook'

    # optimizer
    OptimizerHook = 'OptimizerHook'
    TorchAMPOptimizerHook = 'TorchAMPOptimizerHook'
    ApexAMPOptimizerHook = 'ApexAMPOptimizerHook'
    NoneOptimizerHook = 'NoneOptimizerHook'

    # checkpoint
    CheckpointHook = 'CheckpointHook'
    BestCkptSaverHook = 'BestCkptSaverHook'

    # logger
    TextLoggerHook = 'TextLoggerHook'
    TensorboardHook = 'TensorboardHook'

    IterTimerHook = 'IterTimerHook'
    EvaluationHook = 'EvaluationHook'

    # Compression
    SparsityHook = 'SparsityHook'


class LR_Schedulers(object):
    """learning rate scheduler is defined here

    """
    LinearWarmup = 'LinearWarmup'
    ConstantWarmup = 'ConstantWarmup'
    ExponentialWarmup = 'ExponentialWarmup'


class Datasets(object):
    """ Names for different datasets.
    """
    ClsDataset = 'ClsDataset'
    Face2dKeypointsDataset = 'Face2dKeypointsDataset'
    HumanWholeBodyKeypointDataset = 'HumanWholeBodyKeypointDataset'
    SegDataset = 'SegDataset'
    DetDataset = 'DetDataset'
    DetImagesMixDataset = 'DetImagesMixDataset'
