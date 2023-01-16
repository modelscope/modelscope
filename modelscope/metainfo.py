# Copyright (c) Alibaba, Inc. and its affiliates.


class Models(object):
    """ Names for different models.

        Holds the standard model name to use for identifying different model.
    This should be used to register models.

        Model name should only contain model info but not task info.
    """
    # tinynas models
    tinynas_detection = 'tinynas-detection'
    tinynas_damoyolo = 'tinynas-damoyolo'

    # vision models
    detection = 'detection'
    realtime_object_detection = 'realtime-object-detection'
    realtime_video_object_detection = 'realtime-video-object-detection'
    scrfd = 'scrfd'
    classification_model = 'ClassificationModel'
    easyrobust_model = 'EasyRobustModel'
    bnext = 'bnext'
    nafnet = 'nafnet'
    csrnet = 'csrnet'
    cascade_mask_rcnn_swin = 'cascade_mask_rcnn_swin'
    maskdino_swin = 'maskdino_swin'
    gpen = 'gpen'
    product_retrieval_embedding = 'product-retrieval-embedding'
    body_2d_keypoints = 'body-2d-keypoints'
    body_3d_keypoints = 'body-3d-keypoints'
    crowd_counting = 'HRNetCrowdCounting'
    face_2d_keypoints = 'face-2d-keypoints'
    panoptic_segmentation = 'swinL-panoptic-segmentation'
    r50_panoptic_segmentation = 'r50-panoptic-segmentation'
    image_reid_person = 'passvitb'
    image_inpainting = 'FFTInpainting'
    video_summarization = 'pgl-video-summarization'
    language_guided_video_summarization = 'clip-it-language-guided-video-summarization'
    swinL_semantic_segmentation = 'swinL-semantic-segmentation'
    vitadapter_semantic_segmentation = 'vitadapter-semantic-segmentation'
    text_driven_segmentation = 'text-driven-segmentation'
    newcrfs_depth_estimation = 'newcrfs-depth-estimation'
    panovit_layout_estimation = 'panovit-layout-estimation'
    unifuse_depth_estimation = 'unifuse-depth-estimation'
    dro_resnet18_depth_estimation = 'dro-resnet18-depth-estimation'
    resnet50_bert = 'resnet50-bert'
    referring_video_object_segmentation = 'swinT-referring-video-object-segmentation'
    fer = 'fer'
    fairface = 'fairface'
    retinaface = 'retinaface'
    shop_segmentation = 'shop-segmentation'
    mogface = 'mogface'
    mtcnn = 'mtcnn'
    ulfd = 'ulfd'
    rts = 'rts'
    flir = 'flir'
    arcface = 'arcface'
    facemask = 'facemask'
    flc = 'flc'
    tinymog = 'tinymog'
    video_inpainting = 'video-inpainting'
    human_wholebody_keypoint = 'human-wholebody-keypoint'
    hand_static = 'hand-static'
    face_human_hand_detection = 'face-human-hand-detection'
    face_emotion = 'face-emotion'
    product_segmentation = 'product-segmentation'
    image_body_reshaping = 'image-body-reshaping'
    image_skychange = 'image-skychange'
    video_human_matting = 'video-human-matting'
    video_frame_interpolation = 'video-frame-interpolation'
    video_object_segmentation = 'video-object-segmentation'
    quadtree_attention_image_matching = 'quadtree-attention-image-matching'
    vision_middleware = 'vision-middleware'
    video_stabilization = 'video-stabilization'
    real_basicvsr = 'real-basicvsr'
    rcp_sceneflow_estimation = 'rcp-sceneflow-estimation'
    image_casmvs_depth_estimation = 'image-casmvs-depth-estimation'
    vop_retrieval_model = 'vop-retrieval-model'
    ddcolor = 'ddcolor'
    defrcn = 'defrcn'
    image_face_fusion = 'image-face-fusion'

    # EasyCV models
    yolox = 'YOLOX'
    segformer = 'Segformer'
    hand_2d_keypoints = 'HRNet-Hand2D-Keypoints'
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
    space_T_en = 'space-T-en'
    space_T_cn = 'space-T-cn'
    tcrf = 'transformer-crf'
    token_classification_for_ner = 'token-classification-for-ner'
    tcrf_wseg = 'transformer-crf-for-word-segmentation'
    transformer_softmax = 'transformer-softmax'
    lcrf = 'lstm-crf'
    lcrf_wseg = 'lstm-crf-for-word-segmentation'
    gcnncrf = 'gcnn-crf'
    bart = 'bart'
    gpt2 = 'gpt2'
    gpt3 = 'gpt3'
    gpt_moe = 'gpt-moe'
    gpt_neo = 'gpt-neo'
    plug = 'plug'
    bert_for_ds = 'bert-for-document-segmentation'
    ponet_for_ds = 'ponet-for-document-segmentation'
    ponet = 'ponet'
    T5 = 'T5'
    mglm = 'mglm'
    codegeex = 'codegeex'
    bloom = 'bloom'
    unite = 'unite'
    megatron_bert = 'megatron-bert'
    use = 'user-satisfaction-estimation'

    # audio models
    sambert_hifigan = 'sambert-hifigan'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    speech_kws_fsmn_char_ctc_nearfield = 'speech_kws_fsmn_char_ctc_nearfield'
    speech_mossformer_separation_temporal_8k = 'speech_mossformer_separation_temporal_8k'
    kws_kwsbp = 'kws-kwsbp'
    generic_asr = 'generic-asr'
    wenet_asr = 'wenet-asr'
    generic_itn = 'generic-itn'
    generic_punc = 'generic-punc'
    generic_sv = 'generic-sv'

    # multi-modal models
    ofa = 'ofa'
    clip = 'clip-multi-modal-embedding'
    gemm = 'gemm-generative-multi-modal'
    mplug = 'mplug'
    diffusion = 'diffusion-text-to-image-synthesis'
    multi_stage_diffusion = 'multi-stage-diffusion-text-to-image-synthesis'
    team = 'team-multi-modal-similarity'
    video_clip = 'video-clip-multi-modal-embedding'
    mgeo = 'mgeo'
    vldoc = 'vldoc'
    hitea = 'hitea'

    # science models
    unifold = 'unifold'
    unifold_symmetry = 'unifold-symmetry'


class TaskModels(object):
    # nlp task
    text_classification = 'text-classification'
    token_classification = 'token-classification'
    information_extraction = 'information-extraction'
    fill_mask = 'fill-mask'
    feature_extraction = 'feature-extraction'
    text_generation = 'text-generation'


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
    # text gen
    text_generation = 'text-generation'


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
    image_deblur = 'nafnet-image-deblur'
    person_image_cartoon = 'unet-person-image-cartoon'
    ocr_detection = 'resnet18-ocr-detection'
    table_recognition = 'dla34-table-recognition'
    license_plate_detection = 'resnet18-license-plate-detection'
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
    salient_boudary_detection = 'res2net-salient-detection'
    camouflaged_detection = 'res2net-camouflaged-detection'
    image_classification = 'image-classification'
    face_detection = 'resnet-face-detection-scrfd10gkps'
    face_liveness_ir = 'manual-face-liveness-flir'
    face_liveness_rgb = 'manual-face-liveness-flir'
    card_detection = 'resnet-card-detection-scrfd34gkps'
    ulfd_face_detection = 'manual-face-detection-ulfd'
    tinymog_face_detection = 'manual-face-detection-tinymog'
    facial_expression_recognition = 'vgg19-facial-expression-recognition-fer'
    facial_landmark_confidence = 'manual-facial-landmark-confidence-flcm'
    face_attribute_recognition = 'resnet34-face-attribute-recognition-fairface'
    retina_face_detection = 'resnet50-face-detection-retinaface'
    mog_face_detection = 'resnet101-face-detection-cvpr22papermogface'
    mtcnn_face_detection = 'manual-face-detection-mtcnn'
    live_category = 'live-category'
    general_image_classification = 'vit-base_image-classification_ImageNet-labels'
    daily_image_classification = 'vit-base_image-classification_Dailylife-labels'
    nextvit_small_daily_image_classification = 'nextvit-small_image-classification_Dailylife-labels'
    convnext_base_image_classification_garbage = 'convnext-base_image-classification_garbage'
    bnext_small_image_classification = 'bnext-small_image-classification_ImageNet-labels'
    common_image_classification = 'common-image-classification'
    image_color_enhance = 'csrnet-image-color-enhance'
    virtual_try_on = 'virtual-try-on'
    image_colorization = 'unet-image-colorization'
    image_style_transfer = 'AAMS-style-transfer'
    image_super_resolution = 'rrdb-image-super-resolution'
    face_image_generation = 'gan-face-image-generation'
    product_retrieval_embedding = 'resnet50-product-retrieval-embedding'
    realtime_object_detection = 'cspnet_realtime-object-detection_yolox'
    realtime_video_object_detection = 'cspnet_realtime-video-object-detection_streamyolo'
    face_recognition = 'ir101-face-recognition-cfglint'
    face_recognition_ood = 'ir-face-recognition-ood-rts'
    arc_face_recognition = 'ir50-face-recognition-arcface'
    mask_face_recognition = 'resnet-face-recognition-facemask'
    image_instance_segmentation = 'cascade-mask-rcnn-swin-image-instance-segmentation'
    maskdino_instance_segmentation = 'maskdino-swin-image-instance-segmentation'
    image2image_translation = 'image-to-image-translation'
    live_category = 'live-category'
    video_category = 'video-category'
    ocr_recognition = 'convnextTiny-ocr-recognition'
    image_portrait_enhancement = 'gpen-image-portrait-enhancement'
    image_to_image_generation = 'image-to-image-generation'
    image_object_detection_auto = 'yolox_image-object-detection-auto'
    hand_detection = 'yolox-pai_hand-detection'
    skin_retouching = 'unet-skin-retouching'
    tinynas_classification = 'tinynas-classification'
    easyrobust_classification = 'easyrobust-classification'
    tinynas_detection = 'tinynas-detection'
    crowd_counting = 'hrnet-crowd-counting'
    action_detection = 'ResNetC3D-action-detection'
    video_single_object_tracking = 'ostrack-vitb-video-single-object-tracking'
    video_multi_object_tracking = 'video-multi-object-tracking'
    image_panoptic_segmentation = 'image-panoptic-segmentation'
    image_panoptic_segmentation_easycv = 'image-panoptic-segmentation-easycv'
    video_summarization = 'googlenet_pgl_video_summarization'
    language_guided_video_summarization = 'clip-it-video-summarization'
    image_semantic_segmentation = 'image-semantic-segmentation'
    image_depth_estimation = 'image-depth-estimation'
    indoor_layout_estimation = 'indoor-layout-estimation'
    video_depth_estimation = 'video-depth-estimation'
    panorama_depth_estimation = 'panorama-depth-estimation'
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
    referring_video_object_segmentation = 'referring-video-object-segmentation'
    image_skychange = 'image-skychange'
    video_human_matting = 'video-human-matting'
    vision_middleware_multi_task = 'vision-middleware-multi-task'
    video_frame_interpolation = 'video-frame-interpolation'
    video_object_segmentation = 'video-object-segmentation'
    image_matching = 'image-matching'
    video_stabilization = 'video-stabilization'
    video_super_resolution = 'realbasicvsr-video-super-resolution'
    pointcloud_sceneflow_estimation = 'pointcloud-sceneflow-estimation'
    image_multi_view_depth_estimation = 'image-multi-view-depth-estimation'
    vop_retrieval = 'vop-video-text-retrieval'
    ddcolor_image_colorization = 'ddcolor-image-colorization'
    image_fewshot_detection = 'image-fewshot-detection'
    image_face_fusion = 'image-face-fusion'

    # nlp tasks
    automatic_post_editing = 'automatic-post-editing'
    translation_quality_estimation = 'translation-quality-estimation'
    domain_classification = 'domain-classification'
    sentence_similarity = 'sentence-similarity'
    word_segmentation = 'word-segmentation'
    multilingual_word_segmentation = 'multilingual-word-segmentation'
    word_segmentation_thai = 'word-segmentation-thai'
    part_of_speech = 'part-of-speech'
    named_entity_recognition = 'named-entity-recognition'
    named_entity_recognition_thai = 'named-entity-recognition-thai'
    named_entity_recognition_viet = 'named-entity-recognition-viet'
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
    gpt3_generation = 'gpt3-generation'
    gpt_moe_generation = 'gpt-moe-generation'
    faq_question_answering = 'faq-question-answering'
    conversational_text_to_sql = 'conversational-text-to-sql'
    table_question_answering_pipeline = 'table-question-answering-pipeline'
    sentence_embedding = 'sentence-embedding'
    text_ranking = 'text-ranking'
    mgeo_ranking = 'mgeo-ranking'
    relation_extraction = 'relation-extraction'
    document_segmentation = 'document-segmentation'
    extractive_summarization = 'extractive-summarization'
    feature_extraction = 'feature-extraction'
    mglm_text_summarization = 'mglm-text-summarization'
    codegeex_code_translation = 'codegeex-code-translation'
    codegeex_code_generation = 'codegeex-code-generation'
    translation_en_to_de = 'translation_en_to_de'  # keep it underscore
    translation_en_to_ro = 'translation_en_to_ro'  # keep it underscore
    translation_en_to_fr = 'translation_en_to_fr'  # keep it underscore
    token_classification = 'token-classification'
    translation_evaluation = 'translation-evaluation'
    user_satisfaction_estimation = 'user-satisfaction-estimation'

    # audio tasks
    sambert_hifigan_tts = 'sambert-hifigan-tts'
    speech_dfsmn_aec_psm_16k = 'speech-dfsmn-aec-psm-16k'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    speech_separation = 'speech-separation'
    kws_kwsbp = 'kws-kwsbp'
    asr_inference = 'asr-inference'
    asr_wenet_inference = 'asr-wenet-inference'
    itn_inference = 'itn-inference'
    punc_inference = 'punc-inference'
    sv_inference = 'sv-inference'

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
    ofa_ocr_recognition = 'ofa-ocr-recognition'
    ofa_asr = 'ofa-asr'
    ofa_sudoku = 'ofa-sudoku'
    ofa_text2sql = 'ofa-text2sql'
    video_captioning = 'video-captioning'
    video_question_answering = 'video-question-answering'
    diffusers_stable_diffusion = 'diffusers-stable-diffusion'
    document_vl_embedding = 'document-vl-embedding'
    chinese_stable_diffusion = 'chinese-stable-diffusion'

    # science tasks
    protein_structure = 'unifold-protein-structure'


class Trainers(object):
    """ Names for different trainer.

        Holds the standard trainer name to use for identifying different trainer.
    This should be used to register trainers.

        For a general Trainer, you can use EpochBasedTrainer.
        For a model specific Trainer, you can use ${ModelName}-${Task}-trainer.
    """

    default = 'trainer'
    easycv = 'easycv'
    tinynas_damoyolo = 'tinynas-damoyolo'

    # multi-modal trainers
    clip_multi_modal_embedding = 'clip-multi-modal-embedding'
    ofa = 'ofa'
    mplug = 'mplug'
    mgeo_ranking_trainer = 'mgeo-ranking-trainer'

    # cv trainers
    image_instance_segmentation = 'image-instance-segmentation'
    image_portrait_enhancement = 'image-portrait-enhancement'
    video_summarization = 'video-summarization'
    movie_scene_segmentation = 'movie-scene-segmentation'
    face_detection_scrfd = 'face-detection-scrfd'
    card_detection_scrfd = 'card-detection-scrfd'
    image_inpainting = 'image-inpainting'
    referring_video_object_segmentation = 'referring-video-object-segmentation'
    image_classification_team = 'image-classification-team'
    image_classification = 'image-classification'
    image_fewshot_detection = 'image-fewshot-detection'

    # nlp trainers
    bert_sentiment_analysis = 'bert-sentiment-analysis'
    dialog_modeling_trainer = 'dialog-modeling-trainer'
    dialog_intent_trainer = 'dialog-intent-trainer'
    nlp_base_trainer = 'nlp-base-trainer'
    nlp_veco_trainer = 'nlp-veco-trainer'
    nlp_text_ranking_trainer = 'nlp-text-ranking-trainer'
    text_generation_trainer = 'text-generation-trainer'
    nlp_plug_trainer = 'nlp-plug-trainer'
    gpt3_trainer = 'nlp-gpt3-trainer'
    faq_question_answering_trainer = 'faq-question-answering-trainer'
    gpt_moe_trainer = 'nlp-gpt-moe-trainer'
    table_question_answering_trainer = 'table-question-answering-trainer'

    # audio trainers
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    speech_kws_fsmn_char_ctc_nearfield = 'speech_kws_fsmn_char_ctc_nearfield'
    speech_kantts_trainer = 'speech-kantts-trainer'
    speech_asr_trainer = 'speech-asr-trainer'
    speech_separation = 'speech-separation'


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
    image_denoise_preprocessor = 'image-denoise-preprocessor'
    image_deblur_preprocessor = 'image-deblur-preprocessor'
    object_detection_tinynas_preprocessor = 'object-detection-tinynas-preprocessor'
    image_classification_mmcv_preprocessor = 'image-classification-mmcv-preprocessor'
    image_color_enhance_preprocessor = 'image-color-enhance-preprocessor'
    image_instance_segmentation_preprocessor = 'image-instance-segmentation-preprocessor'
    image_portrait_enhancement_preprocessor = 'image-portrait-enhancement-preprocessor'
    video_summarization_preprocessor = 'video-summarization-preprocessor'
    movie_scene_segmentation_preprocessor = 'movie-scene-segmentation-preprocessor'
    image_classification_bypass_preprocessor = 'image-classification-bypass-preprocessor'
    object_detection_scrfd = 'object-detection-scrfd'
    image_sky_change_preprocessor = 'image-sky-change-preprocessor'

    # nlp preprocessor
    sen_sim_tokenizer = 'sen-sim-tokenizer'
    cross_encoder_tokenizer = 'cross-encoder-tokenizer'
    bert_seq_cls_tokenizer = 'bert-seq-cls-tokenizer'
    text_gen_tokenizer = 'text-gen-tokenizer'
    text2text_gen_preprocessor = 'text2text-gen-preprocessor'
    text_gen_jieba_tokenizer = 'text-gen-jieba-tokenizer'
    text2text_translate_preprocessor = 'text2text-translate-preprocessor'
    token_cls_tokenizer = 'token-cls-tokenizer'
    ner_tokenizer = 'ner-tokenizer'
    thai_ner_tokenizer = 'thai-ner-tokenizer'
    viet_ner_tokenizer = 'viet-ner-tokenizer'
    nli_tokenizer = 'nli-tokenizer'
    sen_cls_tokenizer = 'sen-cls-tokenizer'
    dialog_intent_preprocessor = 'dialog-intent-preprocessor'
    dialog_modeling_preprocessor = 'dialog-modeling-preprocessor'
    dialog_state_tracking_preprocessor = 'dialog-state-tracking-preprocessor'
    sbert_token_cls_tokenizer = 'sbert-token-cls-tokenizer'
    zero_shot_cls_tokenizer = 'zero-shot-cls-tokenizer'
    text_error_correction = 'text-error-correction'
    sentence_embedding = 'sentence-embedding'
    text_ranking = 'text-ranking'
    sequence_labeling_tokenizer = 'sequence-labeling-tokenizer'
    word_segment_text_to_label_preprocessor = 'word-segment-text-to-label-preprocessor'
    thai_wseg_tokenizer = 'thai-wseg-tokenizer'
    fill_mask = 'fill-mask'
    fill_mask_ponet = 'fill-mask-ponet'
    faq_question_answering_preprocessor = 'faq-question-answering-preprocessor'
    conversational_text_to_sql = 'conversational-text-to-sql'
    table_question_answering_preprocessor = 'table-question-answering-preprocessor'
    re_tokenizer = 're-tokenizer'
    document_segmentation = 'document-segmentation'
    feature_extraction = 'feature-extraction'
    mglm_summarization = 'mglm-summarization'
    sentence_piece = 'sentence-piece'
    translation_evaluation = 'translation-evaluation-preprocessor'
    dialog_use_preprocessor = 'dialog-use-preprocessor'

    # audio preprocessor
    linear_aec_fbank = 'linear-aec-fbank'
    text_to_tacotron_symbols = 'text-to-tacotron-symbols'
    wav_to_lists = 'wav-to-lists'
    wav_to_scp = 'wav-to-scp'
    kantts_data_preprocessor = 'kantts-data-preprocessor'

    # multi-modal preprocessor
    ofa_tasks_preprocessor = 'ofa-tasks-preprocessor'
    clip_preprocessor = 'clip-preprocessor'
    mplug_tasks_preprocessor = 'mplug-tasks-preprocessor'
    mgeo_ranking = 'mgeo-ranking'
    vldoc_preprocessor = 'vldoc-preprocessor'
    hitea_tasks_preprocessor = 'hitea-tasks-preprocessor'

    # science preprocessor
    unifold_preprocessor = 'unifold-preprocessor'


class Metrics(object):
    """ Names for different metrics.
    """

    # accuracy
    accuracy = 'accuracy'

    multi_average_precision = 'mAP'
    audio_noise_metric = 'audio-noise-metric'
    PPL = 'ppl'

    # text gen
    BLEU = 'bleu'

    # metrics for image denoise task
    image_denoise_metric = 'image-denoise-metric'
    # metrics for video frame-interpolation task
    video_frame_interpolation_metric = 'video-frame-interpolation-metric'
    # metrics for real-world video super-resolution task
    video_super_resolution_metric = 'video-super-resolution-metric'

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
    # metric for ocr
    NED = 'ned'
    # metric for cross-modal retrieval
    inbatch_recall = 'inbatch_recall'
    # metric for referring-video-object-segmentation task
    referring_video_object_segmentation_metric = 'referring-video-object-segmentation-metric'
    # metric for video stabilization task
    video_stabilization_metric = 'video-stabilization-metric'


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

    # CLIP logit_scale clamp
    ClipClampLogitScaleHook = 'ClipClampLogitScaleHook'

    # train
    EarlyStopHook = 'EarlyStopHook'
    DeepspeedHook = 'DeepspeedHook'


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
    Face2dKeypointsDataset = 'FaceKeypointDataset'
    HandCocoWholeBodyDataset = 'HandCocoWholeBodyDataset'
    HumanWholeBodyKeypointDataset = 'WholeBodyCocoTopDownDataset'
    SegDataset = 'SegDataset'
    DetDataset = 'DetDataset'
    DetImagesMixDataset = 'DetImagesMixDataset'
    PanopticDataset = 'PanopticDataset'
    PairedDataset = 'PairedDataset'
