# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.utils.constant import Fields, Tasks


class Models(object):
    """ Names for different models.

        Holds the standard model name to use for identifying different model.
    This should be used to register models.

        Model name should only contain model information but not task information.
    """
    # tinynas models
    tinynas_detection = 'tinynas-detection'
    tinynas_damoyolo = 'tinynas-damoyolo'
    # vision models
    detection = 'detection'
    mask_scoring = 'MaskScoring'
    image_restoration = 'image-restoration'
    realtime_object_detection = 'realtime-object-detection'
    realtime_video_object_detection = 'realtime-video-object-detection'
    scrfd = 'scrfd'
    depe = 'depe'
    classification_model = 'ClassificationModel'
    easyrobust_model = 'EasyRobustModel'
    bnext = 'bnext'
    yolopv2 = 'yolopv2'
    nafnet = 'nafnet'
    csrnet = 'csrnet'
    adaint = 'adaint'
    deeplpfnet = 'deeplpfnet'
    rrdb = 'rrdb'
    cascade_mask_rcnn_swin = 'cascade_mask_rcnn_swin'
    maskdino_swin = 'maskdino_swin'
    gpen = 'gpen'
    product_retrieval_embedding = 'product-retrieval-embedding'
    body_2d_keypoints = 'body-2d-keypoints'
    body_3d_keypoints = 'body-3d-keypoints'
    body_3d_keypoints_hdformer = 'hdformer'
    crowd_counting = 'HRNetCrowdCounting'
    face_2d_keypoints = 'face-2d-keypoints'
    panoptic_segmentation = 'swinL-panoptic-segmentation'
    r50_panoptic_segmentation = 'r50-panoptic-segmentation'
    image_reid_person = 'passvitb'
    image_inpainting = 'FFTInpainting'
    image_paintbyexample = 'Stablediffusion-Paintbyexample'
    video_summarization = 'pgl-video-summarization'
    video_panoptic_segmentation = 'swinb-video-panoptic-segmentation'
    video_instance_segmentation = 'swinb-video-instance-segmentation'
    language_guided_video_summarization = 'clip-it-language-guided-video-summarization'
    swinL_semantic_segmentation = 'swinL-semantic-segmentation'
    vitadapter_semantic_segmentation = 'vitadapter-semantic-segmentation'
    text_driven_segmentation = 'text-driven-segmentation'
    newcrfs_depth_estimation = 'newcrfs-depth-estimation'
    omnidata_normal_estimation = 'omnidata-normal-estimation'
    panovit_layout_estimation = 'panovit-layout-estimation'
    unifuse_depth_estimation = 'unifuse-depth-estimation'
    s2net_depth_estimation = 's2net-depth-estimation'
    dro_resnet18_depth_estimation = 'dro-resnet18-depth-estimation'
    raft_dense_optical_flow_estimation = 'raft-dense-optical-flow-estimation'
    resnet50_bert = 'resnet50-bert'
    referring_video_object_segmentation = 'swinT-referring-video-object-segmentation'
    fer = 'fer'
    fairface = 'fairface'
    retinaface = 'retinaface'
    damofd = 'damofd'
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
    human_reconstruction = 'human-reconstruction'
    text_texture_generation = 'text-texture-generation'
    video_frame_interpolation = 'video-frame-interpolation'
    video_object_segmentation = 'video-object-segmentation'
    video_deinterlace = 'video-deinterlace'
    quadtree_attention_image_matching = 'quadtree-attention-image-matching'
    loftr_image_local_feature_matching = 'loftr-image-local-feature-matching'
    lightglue_image_matching = 'lightglue-image-matching'
    vision_middleware = 'vision-middleware'
    vidt = 'vidt'
    video_stabilization = 'video-stabilization'
    real_basicvsr = 'real-basicvsr'
    rcp_sceneflow_estimation = 'rcp-sceneflow-estimation'
    image_casmvs_depth_estimation = 'image-casmvs-depth-estimation'
    image_geomvsnet_depth_estimation = 'image-geomvsnet-depth-estimation'
    vop_retrieval_model = 'vop-retrieval-model'
    vop_retrieval_model_se = 'vop-retrieval-model-se'
    ddcolor = 'ddcolor'
    image_probing_model = 'image-probing-model'
    defrcn = 'defrcn'
    image_face_fusion = 'image-face-fusion'
    content_check = 'content-check'
    open_vocabulary_detection_vild = 'open-vocabulary-detection-vild'
    ecbsr = 'ecbsr'
    msrresnet_lite = 'msrresnet-lite'
    object_detection_3d = 'object_detection_3d'
    ddpm = 'ddpm'
    ocr_recognition = 'OCRRecognition'
    ocr_detection = 'OCRDetection'
    lineless_table_recognition = 'LoreModel'
    image_quality_assessment_mos = 'image-quality-assessment-mos'
    image_quality_assessment_man = 'image-quality-assessment-man'
    image_quality_assessment_degradation = 'image-quality-assessment-degradation'
    m2fp = 'm2fp'
    nerf_recon_acc = 'nerf-recon-acc'
    nerf_recon_4k = 'nerf-recon-4k'
    nerf_recon_vq_compression = 'nerf-recon-vq-compression'
    surface_recon_common = 'surface-recon-common'
    bts_depth_estimation = 'bts-depth-estimation'
    vision_efficient_tuning = 'vision-efficient-tuning'
    bad_image_detecting = 'bad-image-detecting'
    controllable_image_generation = 'controllable-image-generation'
    longshortnet = 'longshortnet'
    fastinst = 'fastinst'
    pedestrian_attribute_recognition = 'pedestrian-attribute-recognition'
    image_try_on = 'image-try-on'
    human_image_generation = 'human-image-generation'
    image_view_transform = 'image-view-transform'
    image_control_3d_portrait = 'image-control-3d-portrait'
    rife = 'rife'
    anydoor = 'anydoor'
    self_supervised_depth_completion = 'self-supervised-depth-completion'

    # nlp models
    bert = 'bert'
    palm = 'palm-v2'
    structbert = 'structbert'
    deberta_v2 = 'deberta_v2'
    veco = 'veco'
    translation = 'csanmt-translation'
    canmt = 'canmt'
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
    polylm = 'polylm'
    T5 = 'T5'
    mglm = 'mglm'
    codegeex = 'codegeex'
    glm130b = 'glm130b'
    bloom = 'bloom'
    unite = 'unite'
    megatron_bert = 'megatron-bert'
    use = 'user-satisfaction-estimation'
    fid_plug = 'fid-plug'
    fid_T5 = 'fid-T5'
    lstm = 'lstm'
    xlm_roberta = 'xlm-roberta'
    transformers = 'transformers'
    plug_mental = 'plug-mental'
    doc2bot = 'doc2bot'
    peer = 'peer'
    llama = 'llama'
    llama2 = 'llama2'
    chatglm_6b = 'chatglm6b'
    chatglm2_6b = 'chatglm2-6b'
    qwen_7b = 'qwen-7b'

    # audio models
    sambert_hifigan = 'sambert-hifigan'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_ans = 'speech_dfsmn_ans'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    speech_dfsmn_kws_char_farfield_iot = 'speech_dfsmn_kws_char_farfield_iot'
    speech_kws_fsmn_char_ctc_nearfield = 'speech_kws_fsmn_char_ctc_nearfield'
    speech_mossformer_separation_temporal_8k = 'speech_mossformer_separation_temporal_8k'
    speech_mossformer2_separation_temporal_8k = 'speech_mossformer2_separation_temporal_8k'
    kws_kwsbp = 'kws-kwsbp'
    generic_asr = 'generic-asr'
    wenet_asr = 'wenet-asr'
    generic_itn = 'generic-itn'
    generic_punc = 'generic-punc'
    generic_sv = 'generic-sv'
    tdnn_sv = 'tdnn-sv'
    ecapa_tdnn_sv = 'ecapa-tdnn-sv'
    campplus_sv = 'cam++-sv'
    eres2net_sv = 'eres2net-sv'
    eres2netv2_sv = 'eres2netv2-sv'
    resnet_sv = 'resnet-sv'
    res2net_sv = 'res2net-sv'
    eres2net_aug_sv = 'eres2net-aug-sv'
    scl_sd = 'scl-sd'
    scl_sd_xvector = 'scl-sd-xvector'
    campplus_lre = 'cam++-lre'
    eres2net_lre = 'eres2net-lre'
    cluster_backend = 'cluster-backend'
    rdino_tdnn_sv = 'rdino_ecapa-tdnn-sv'
    sdpn_sv = 'sdpn_ecapa-sv'
    generic_lm = 'generic-lm'
    audio_quantization = 'audio-quantization'
    laura_codec = 'laura-codec'
    funasr = 'funasr'

    # multi-modal models
    ofa = 'ofa'
    clip = 'clip-multi-modal-embedding'
    gemm = 'gemm-generative-multi-modal'
    rleg = 'rleg-generative-multi-modal'
    mplug = 'mplug'
    diffusion = 'diffusion-text-to-image-synthesis'
    multi_stage_diffusion = 'multi-stage-diffusion-text-to-image-synthesis'
    video_synthesis = 'latent-text-to-video-synthesis'
    team = 'team-multi-modal-similarity'
    video_clip = 'video-clip-multi-modal-embedding'
    prost = 'prost-clip-text-video-retrieval'
    mgeo = 'mgeo'
    vldoc = 'vldoc'
    hitea = 'hitea'
    soonet = 'soonet'
    efficient_diffusion_tuning = 'efficient-diffusion-tuning'
    cones2_inference = 'cones2-inference'
    mplug_owl = 'mplug-owl'

    clip_interrogator = 'clip-interrogator'
    stable_diffusion = 'stable-diffusion'
    stable_diffusion_xl = 'stable-diffusion-xl'
    videocomposer = 'videocomposer'
    text_to_360panorama_image = 'text-to-360panorama-image'
    image_to_video_model = 'image-to-video-model'
    video_to_video_model = 'video-to-video-model'

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
    text_ranking = 'text-ranking'
    machine_reading_comprehension = 'machine-reading-comprehension'


class Heads(object):
    # nlp heads

    # text cls
    text_classification = 'text-classification'
    # fill mask
    fill_mask = 'fill-mask'
    bert_mlm = 'bert-mlm'
    roberta_mlm = 'roberta-mlm'
    xlm_roberta_mlm = 'xlm-roberta-mlm'
    # token cls
    token_classification = 'token-classification'
    # extraction
    information_extraction = 'information-extraction'
    # text gen
    text_generation = 'text-generation'
    # text ranking
    text_ranking = 'text-ranking'
    # crf
    lstm_crf = 'lstm-crf'
    transformer_crf = 'transformer-crf'


class Pipelines(object):
    """ Names for different pipelines.

        Holds the standard pipline name to use for identifying different pipeline.
    This should be used to register pipelines.

        For pipeline which support different models and implements the common function, we
    should use task name for this pipeline.
        For pipeline which suuport only one model, we should use ${Model}-${Task} as its name.
    """
    pipeline_template = 'pipeline-template'
    # vision tasks
    portrait_matting = 'unet-image-matting'
    universal_matting = 'unet-universal-matting'
    image_denoise = 'nafnet-image-denoise'
    image_deblur = 'nafnet-image-deblur'
    image_editing = 'masactrl-image-editing'
    freeu_stable_diffusion_text2image = 'freeu-stable-diffusion-text2image'
    person_image_cartoon = 'unet-person-image-cartoon'
    ocr_detection = 'resnet18-ocr-detection'
    table_recognition = 'dla34-table-recognition'
    lineless_table_recognition = 'lore-lineless-table-recognition'
    license_plate_detection = 'resnet18-license-plate-detection'
    card_detection_correction = 'resnet18-card-detection-correction'
    action_recognition = 'TAdaConv_action-recognition'
    animal_recognition = 'resnet101-animal-recognition'
    general_recognition = 'resnet101-general-recognition'
    cmdssl_video_embedding = 'cmdssl-r2p1d_video_embedding'
    hicossl_video_embedding = 'hicossl-s3dg-video_embedding'
    body_2d_keypoints = 'hrnetv2w32_body-2d-keypoints_image'
    body_3d_keypoints = 'canonical_body-3d-keypoints_video'
    hand_2d_keypoints = 'hrnetv2w18_hand-2d-keypoints_image'
    human_detection = 'resnet18-human-detection'
    tbs_detection = 'tbs-detection'
    object_detection = 'vit-object-detection'
    abnormal_object_detection = 'abnormal-object-detection'
    face_2d_keypoints = 'mobilenet_face-2d-keypoints_alignment'
    salient_detection = 'u2net-salient-detection'
    salient_boudary_detection = 'res2net-salient-detection'
    camouflaged_detection = 'res2net-camouflaged-detection'
    image_demoire = 'uhdm-image-demoireing'
    image_classification = 'image-classification'
    face_detection = 'resnet-face-detection-scrfd10gkps'
    face_liveness_ir = 'manual-face-liveness-flir'
    face_liveness_rgb = 'manual-face-liveness-flir'
    face_liveness_xc = 'manual-face-liveness-flxc'
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
    yolopv2_image_driving_percetion_bdd100k = 'yolopv2_image-driving-percetion_bdd100k'
    common_image_classification = 'common-image-classification'
    image_color_enhance = 'csrnet-image-color-enhance'
    adaint_image_color_enhance = 'adaint-image-color-enhance'
    deeplpf_image_color_enhance = 'deeplpf-image-color-enhance'
    virtual_try_on = 'virtual-try-on'
    image_colorization = 'unet-image-colorization'
    image_style_transfer = 'AAMS-style-transfer'
    image_super_resolution = 'rrdb-image-super-resolution'
    image_super_resolution_pasd = 'image-super-resolution-pasd'
    image_debanding = 'rrdb-image-debanding'
    face_image_generation = 'gan-face-image-generation'
    product_retrieval_embedding = 'resnet50-product-retrieval-embedding'
    realtime_video_object_detection = 'cspnet_realtime-video-object-detection_streamyolo'
    face_recognition = 'ir101-face-recognition-cfglint'
    face_recognition_ood = 'ir-face-recognition-ood-rts'
    face_quality_assessment = 'manual-face-quality-assessment-fqa'
    face_recognition_ood = 'ir-face-recognition-rts'
    face_recognition_onnx_ir = 'manual-face-recognition-frir'
    face_recognition_onnx_fm = 'manual-face-recognition-frfm'
    arc_face_recognition = 'ir50-face-recognition-arcface'
    mask_face_recognition = 'resnet-face-recognition-facemask'
    content_check = 'resnet50-image-classification-cc'
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
    face_reconstruction = 'resnet50-face-reconstruction'
    head_reconstruction = 'HRN-head-reconstruction'
    text_to_head = 'HRN-text-to-head'
    tinynas_classification = 'tinynas-classification'
    easyrobust_classification = 'easyrobust-classification'
    tinynas_detection = 'tinynas-detection'
    crowd_counting = 'hrnet-crowd-counting'
    action_detection = 'ResNetC3D-action-detection'
    video_single_object_tracking = 'ostrack-vitb-video-single-object-tracking'
    video_single_object_tracking_procontext = 'procontext-vitb-video-single-object-tracking'
    video_multi_object_tracking = 'video-multi-object-tracking'
    image_panoptic_segmentation = 'image-panoptic-segmentation'
    video_summarization = 'googlenet_pgl_video_summarization'
    language_guided_video_summarization = 'clip-it-video-summarization'
    image_semantic_segmentation = 'image-semantic-segmentation'
    image_depth_estimation = 'image-depth-estimation'
    image_normal_estimation = 'image-normal-estimation'
    indoor_layout_estimation = 'indoor-layout-estimation'
    image_local_feature_matching = 'image-local-feature-matching'
    video_depth_estimation = 'video-depth-estimation'
    panorama_depth_estimation = 'panorama-depth-estimation'
    panorama_depth_estimation_s2net = 'panorama-depth-estimation-s2net'
    dense_optical_flow_estimation = 'dense-optical-flow-estimation'
    image_reid_person = 'passvitb-image-reid-person'
    image_inpainting = 'fft-inpainting'
    image_paintbyexample = 'stablediffusion-paintbyexample'
    image_inpainting_sdv2 = 'image-inpainting-sdv2'
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
    human_reconstruction = 'human-reconstruction'
    text_texture_generation = 'text-texture-generation'
    vision_middleware_multi_task = 'vision-middleware-multi-task'
    vidt = 'vidt'
    video_frame_interpolation = 'video-frame-interpolation'
    video_object_segmentation = 'video-object-segmentation'
    video_deinterlace = 'video-deinterlace'
    image_matching = 'image-matching'
    image_matching_fast = 'image-matching-fast'
    video_stabilization = 'video-stabilization'
    video_super_resolution = 'realbasicvsr-video-super-resolution'
    pointcloud_sceneflow_estimation = 'pointcloud-sceneflow-estimation'
    image_multi_view_depth_estimation = 'image-multi-view-depth-estimation'
    video_panoptic_segmentation = 'video-panoptic-segmentation'
    video_instance_segmentation = 'video-instance-segmentation'
    vop_retrieval = 'vop-video-text-retrieval'
    vop_retrieval_se = 'vop-video-text-retrieval-se'
    ddcolor_image_colorization = 'ddcolor-image-colorization'
    image_structured_model_probing = 'image-structured-model-probing'
    image_fewshot_detection = 'image-fewshot-detection'
    image_face_fusion = 'image-face-fusion'
    open_vocabulary_detection_vild = 'open-vocabulary-detection-vild'
    ddpm_image_semantic_segmentation = 'ddpm-image-semantic-segmentation'
    video_colorization = 'video-colorization'
    motion_generattion = 'mdm-motion-generation'
    mobile_image_super_resolution = 'mobile-image-super-resolution'
    image_human_parsing = 'm2fp-image-human-parsing'
    object_detection_3d_depe = 'object-detection-3d-depe'
    nerf_recon_acc = 'nerf-recon-acc'
    nerf_recon_4k = 'nerf-recon-4k'
    nerf_recon_vq_compression = 'nerf-recon-vq-compression'
    surface_recon_common = 'surface-recon-common'
    bad_image_detecting = 'bad-image-detecting'
    controllable_image_generation = 'controllable-image-generation'
    fast_instance_segmentation = 'fast-instance-segmentation'
    image_quality_assessment_mos = 'image-quality-assessment-mos'
    image_quality_assessment_man = 'image-quality-assessment-man'
    image_quality_assessment_degradation = 'image-quality-assessment-degradation'
    vision_efficient_tuning = 'vision-efficient-tuning'
    image_bts_depth_estimation = 'image-bts-depth-estimation'
    image_depth_estimation_marigold = 'image-depth-estimation-marigold'
    pedestrian_attribute_recognition = 'resnet50_pedestrian-attribute-recognition_image'
    text_to_360panorama_image = 'text-to-360panorama-image'
    image_try_on = 'image-try-on'
    human_image_generation = 'human-image-generation'
    human3d_render = 'human3d-render'
    human3d_animation = 'human3d-animation'
    image_view_transform = 'image-view-transform'
    image_control_3d_portrait = 'image-control-3d-portrait'
    rife_video_frame_interpolation = 'rife-video-frame-interpolation'
    anydoor = 'anydoor'
    image_to_3d = 'image-to-3d'
    self_supervised_depth_completion = 'self-supervised-depth-completion'

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
    fid_dialogue = 'fid-dialogue'
    text2text_generation = 'text2text-generation'
    sentiment_analysis = 'sentiment-analysis'
    sentiment_classification = 'sentiment-classification'
    text_classification = 'text-classification'
    fill_mask = 'fill-mask'
    fill_mask_ponet = 'fill-mask-ponet'
    csanmt_translation = 'csanmt-translation'
    canmt_translation = 'canmt-translation'
    interactive_translation = 'interactive-translation'
    nli = 'nli'
    dialog_intent_prediction = 'dialog-intent-prediction'
    dialog_modeling = 'dialog-modeling'
    dialog_state_tracking = 'dialog-state-tracking'
    zero_shot_classification = 'zero-shot-classification'
    text_error_correction = 'text-error-correction'
    word_alignment = 'word-alignment'
    plug_generation = 'plug-generation'
    gpt3_generation = 'gpt3-generation'
    polylm_text_generation = 'polylm-text-generation'
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
    glm130b_text_generation = 'glm130b-text-generation'
    translation_en_to_de = 'translation_en_to_de'  # keep it underscore
    translation_en_to_ro = 'translation_en_to_ro'  # keep it underscore
    translation_en_to_fr = 'translation_en_to_fr'  # keep it underscore
    token_classification = 'token-classification'
    translation_evaluation = 'translation-evaluation'
    user_satisfaction_estimation = 'user-satisfaction-estimation'
    siamese_uie = 'siamese-uie'
    document_grounded_dialog_retrieval = 'document-grounded-dialog-retrieval'
    document_grounded_dialog_rerank = 'document-grounded-dialog-rerank'
    document_grounded_dialog_generate = 'document-grounded-dialog-generate'
    language_identification = 'language_identification'
    machine_reading_comprehension_for_ner = 'machine-reading-comprehension-for-ner'
    llm = 'llm'

    # audio tasks
    sambert_hifigan_tts = 'sambert-hifigan-tts'
    speech_dfsmn_aec_psm_16k = 'speech-dfsmn-aec-psm-16k'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_ans_psm_48k_causal = 'speech_dfsmn_ans_psm_48k_causal'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    speech_separation = 'speech-separation'
    kws_kwsbp = 'kws-kwsbp'
    asr_wenet_inference = 'asr-wenet-inference'
    itn_inference = 'itn-inference'
    speaker_diarization_inference = 'speaker-diarization-inference'
    vad_inference = 'vad-inference'
    funasr_speech_separation = 'funasr-speech-separation'
    speaker_verification = 'speaker-verification'
    speaker_verification_tdnn = 'speaker-verification-tdnn'
    speaker_verification_rdino = 'speaker-verification-rdino'
    speaker_verification_sdpn = 'speaker-verification-sdpn'
    speaker_verification_eres2net = 'speaker-verification-eres2net'
    speaker_verification_eres2netv2 = 'speaker-verification-eres2netv2'
    speaker_verification_resnet = 'speaker-verification-resnet'
    speaker_verification_res2net = 'speaker-verification-res2net'
    speech_language_recognition = 'speech-language-recognition'
    speech_language_recognition_eres2net = 'speech-language-recognition-eres2net'
    speaker_change_locating = 'speaker-change-locating'
    speaker_diarization_dialogue_detection = 'speaker-diarization-dialogue-detection'
    speaker_diarization_semantic_speaker_turn_detection = 'speaker-diarization-semantic-speaker-turn-detection'
    segmentation_clustering = 'segmentation-clustering'
    lm_inference = 'language-score-prediction'
    speech_timestamp_inference = 'speech-timestamp-inference'
    audio_quantization = 'audio-quantization'
    audio_quantization_inference = 'audio-quantization-inference'
    laura_codec_tts_inference = 'laura-codec-tts-inference'

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
    prost_text_video_retrieval = 'prost-text-video-retrieval'
    videocomposer = 'videocomposer'
    image_text_retrieval = 'image-text-retrieval'
    ofa_ocr_recognition = 'ofa-ocr-recognition'
    ofa_asr = 'ofa-asr'
    ofa_sudoku = 'ofa-sudoku'
    ofa_text2sql = 'ofa-text2sql'
    video_captioning = 'video-captioning'
    video_question_answering = 'video-question-answering'
    diffusers_stable_diffusion = 'diffusers-stable-diffusion'
    disco_guided_diffusion = 'disco_guided_diffusion'
    document_vl_embedding = 'document-vl-embedding'
    chinese_stable_diffusion = 'chinese-stable-diffusion'
    cones2_inference = 'cones2-inference'
    text_to_video_synthesis = 'latent-text-to-video-synthesis'  # latent-text-to-video-synthesis
    gridvlp_multi_modal_classification = 'gridvlp-multi-modal-classification'
    gridvlp_multi_modal_embedding = 'gridvlp-multi-modal-embedding'
    soonet_video_temporal_grounding = 'soonet-video-temporal-grounding'
    efficient_diffusion_tuning = 'efficient-diffusion-tuning'
    multimodal_dialogue = 'multimodal-dialogue'
    llama2_text_generation_pipeline = 'llama2-text-generation-pipeline'
    llama2_text_generation_chat_pipeline = 'llama2-text-generation-chat-pipeline'
    image_to_video_task_pipeline = 'image-to-video-task-pipeline'
    video_to_video_pipeline = 'video-to-video-pipeline'

    # science tasks
    protein_structure = 'unifold-protein-structure'

    # funasr task
    funasr_pipeline = 'funasr-pipeline'


DEFAULT_MODEL_FOR_PIPELINE = {
    # TaskName: (pipeline_module_name, model_repo)
    Tasks.sentence_embedding:
    (Pipelines.sentence_embedding,
     'damo/nlp_corom_sentence-embedding_english-base'),
    Tasks.text_ranking: (Pipelines.mgeo_ranking,
                         'damo/mgeo_address_ranking_chinese_base'),
    Tasks.text_ranking: (Pipelines.text_ranking,
                         'damo/nlp_corom_passage-ranking_english-base'),
    Tasks.word_segmentation:
    (Pipelines.word_segmentation,
     'damo/nlp_structbert_word-segmentation_chinese-base'),
    Tasks.part_of_speech: (Pipelines.part_of_speech,
                           'damo/nlp_structbert_part-of-speech_chinese-base'),
    Tasks.token_classification:
    (Pipelines.part_of_speech,
     'damo/nlp_structbert_part-of-speech_chinese-base'),
    Tasks.named_entity_recognition:
    (Pipelines.named_entity_recognition,
     'damo/nlp_raner_named-entity-recognition_chinese-base-news'),
    Tasks.relation_extraction:
    (Pipelines.relation_extraction,
     'damo/nlp_bert_relation-extraction_chinese-base'),
    Tasks.information_extraction:
    (Pipelines.relation_extraction,
     'damo/nlp_bert_relation-extraction_chinese-base'),
    Tasks.sentence_similarity:
    (Pipelines.sentence_similarity,
     'damo/nlp_structbert_sentence-similarity_chinese-base'),
    Tasks.competency_aware_translation:
    (Pipelines.canmt_translation, 'damo/nlp_canmt_translation_zh2en_large'),
    Tasks.translation: (Pipelines.csanmt_translation,
                        'damo/nlp_csanmt_translation_zh2en'),
    Tasks.nli: (Pipelines.nli, 'damo/nlp_structbert_nli_chinese-base'),
    Tasks.sentiment_classification:
    (Pipelines.sentiment_classification,
     'damo/nlp_structbert_sentiment-classification_chinese-base'
     ),  # TODO: revise back after passing the pr
    Tasks.portrait_matting: (Pipelines.portrait_matting,
                             'damo/cv_unet_image-matting'),
    Tasks.universal_matting: (Pipelines.universal_matting,
                              'damo/cv_unet_universal-matting'),
    Tasks.human_detection: (Pipelines.human_detection,
                            'damo/cv_resnet18_human-detection'),
    Tasks.image_object_detection: (Pipelines.object_detection,
                                   'damo/cv_vit_object-detection_coco'),
    Tasks.image_denoising: (Pipelines.image_denoise,
                            'damo/cv_nafnet_image-denoise_sidd'),
    Tasks.image_deblurring: (Pipelines.image_deblur,
                             'damo/cv_nafnet_image-deblur_gopro'),
    Tasks.image_editing: (Pipelines.image_editing,
                          'damo/cv_masactrl_image-editing'),
    Tasks.video_stabilization: (Pipelines.video_stabilization,
                                'damo/cv_dut-raft_video-stabilization_base'),
    Tasks.video_super_resolution:
    (Pipelines.video_super_resolution,
     'damo/cv_realbasicvsr_video-super-resolution_videolq'),
    Tasks.text_classification:
    (Pipelines.sentiment_classification,
     'damo/nlp_structbert_sentiment-classification_chinese-base'),
    Tasks.text_generation: (Pipelines.text_generation,
                            'damo/nlp_palm2.0_text-generation_chinese-base'),
    Tasks.zero_shot_classification:
    (Pipelines.zero_shot_classification,
     'damo/nlp_structbert_zero-shot-classification_chinese-base'),
    Tasks.task_oriented_conversation: (Pipelines.dialog_modeling,
                                       'damo/nlp_space_dialog-modeling'),
    Tasks.dialog_state_tracking: (Pipelines.dialog_state_tracking,
                                  'damo/nlp_space_dialog-state-tracking'),
    Tasks.table_question_answering:
    (Pipelines.table_question_answering_pipeline,
     'damo/nlp-convai-text2sql-pretrain-cn'),
    Tasks.document_grounded_dialog_generate:
    (Pipelines.document_grounded_dialog_generate,
     'DAMO_ConvAI/nlp_convai_generation_pretrain'),
    Tasks.document_grounded_dialog_rerank:
    (Pipelines.document_grounded_dialog_rerank,
     'damo/nlp_convai_rerank_pretrain'),
    Tasks.document_grounded_dialog_retrieval:
    (Pipelines.document_grounded_dialog_retrieval,
     'DAMO_ConvAI/nlp_convai_retrieval_pretrain'),
    Tasks.text_error_correction:
    (Pipelines.text_error_correction,
     'damo/nlp_bart_text-error-correction_chinese'),
    Tasks.image_captioning: (Pipelines.image_captioning,
                             'damo/ofa_image-caption_coco_large_en'),
    Tasks.video_captioning:
    (Pipelines.video_captioning,
     'damo/multi-modal_hitea_video-captioning_base_en'),
    Tasks.image_portrait_stylization:
    (Pipelines.person_image_cartoon,
     'damo/cv_unet_person-image-cartoon_compound-models'),
    Tasks.ocr_detection: (Pipelines.ocr_detection,
                          'damo/cv_resnet18_ocr-detection-line-level_damo'),
    Tasks.table_recognition:
    (Pipelines.table_recognition,
     'damo/cv_dla34_table-structure-recognition_cycle-centernet'),
    Tasks.lineless_table_recognition:
    (Pipelines.lineless_table_recognition,
     'damo/cv_resnet-transformer_table-structure-recognition_lore'),
    Tasks.document_vl_embedding:
    (Pipelines.document_vl_embedding,
     'damo/multi-modal_convnext-roberta-base_vldoc-embedding'),
    Tasks.license_plate_detection:
    (Pipelines.license_plate_detection,
     'damo/cv_resnet18_license-plate-detection_damo'),
    Tasks.card_detection_correction: (Pipelines.card_detection_correction,
                                      'damo/cv_resnet18_card_correction'),
    Tasks.fill_mask: (Pipelines.fill_mask, 'damo/nlp_veco_fill-mask-large'),
    Tasks.feature_extraction: (Pipelines.feature_extraction,
                               'damo/pert_feature-extraction_base-test'),
    Tasks.action_recognition: (Pipelines.action_recognition,
                               'damo/cv_TAdaConv_action-recognition'),
    Tasks.action_detection: (Pipelines.action_detection,
                             'damo/cv_ResNetC3D_action-detection_detection2d'),
    Tasks.live_category: (Pipelines.live_category,
                          'damo/cv_resnet50_live-category'),
    Tasks.video_category: (Pipelines.video_category,
                           'damo/cv_resnet50_video-category'),
    Tasks.multi_modal_embedding: (Pipelines.multi_modal_embedding,
                                  'damo/multi-modal_clip-vit-base-patch16_zh'),
    Tasks.generative_multi_modal_embedding:
    (Pipelines.generative_multi_modal_embedding,
     'damo/multi-modal_gemm-vit-large-patch14_generative-multi-modal-embedding'
     ),
    Tasks.multi_modal_similarity:
    (Pipelines.multi_modal_similarity,
     'damo/multi-modal_team-vit-large-patch14_multi-modal-similarity'),
    Tasks.visual_question_answering:
    (Pipelines.visual_question_answering,
     'damo/mplug_visual-question-answering_coco_large_en'),
    Tasks.video_question_answering:
    (Pipelines.video_question_answering,
     'damo/multi-modal_hitea_video-question-answering_base_en'),
    Tasks.video_embedding: (Pipelines.cmdssl_video_embedding,
                            'damo/cv_r2p1d_video_embedding'),
    Tasks.text_to_image_synthesis:
    (Pipelines.text_to_image_synthesis,
     'damo/cv_diffusion_text-to-image-synthesis_tiny'),
    Tasks.text_to_video_synthesis: (Pipelines.text_to_video_synthesis,
                                    'damo/text-to-video-synthesis'),
    Tasks.body_2d_keypoints: (Pipelines.body_2d_keypoints,
                              'damo/cv_hrnetv2w32_body-2d-keypoints_image'),
    Tasks.body_3d_keypoints: (Pipelines.body_3d_keypoints,
                              'damo/cv_canonical_body-3d-keypoints_video'),
    Tasks.hand_2d_keypoints:
    (Pipelines.hand_2d_keypoints,
     'damo/cv_hrnetw18_hand-pose-keypoints_coco-wholebody'),
    Tasks.card_detection: (Pipelines.card_detection,
                           'damo/cv_resnet_carddetection_scrfd34gkps'),
    Tasks.content_check: (Pipelines.content_check,
                          'damo/cv_resnet50_content-check_cc'),
    Tasks.face_detection:
    (Pipelines.mog_face_detection,
     'damo/cv_resnet101_face-detection_cvpr22papermogface'),
    Tasks.face_liveness: (Pipelines.face_liveness_ir,
                          'damo/cv_manual_face-liveness_flir'),
    Tasks.face_recognition: (Pipelines.face_recognition,
                             'damo/cv_ir101_facerecognition_cfglint'),
    Tasks.facial_expression_recognition:
    (Pipelines.facial_expression_recognition,
     'damo/cv_vgg19_facial-expression-recognition_fer'),
    Tasks.face_attribute_recognition:
    (Pipelines.face_attribute_recognition,
     'damo/cv_resnet34_face-attribute-recognition_fairface'),
    Tasks.face_2d_keypoints: (Pipelines.face_2d_keypoints,
                              'damo/cv_mobilenet_face-2d-keypoints_alignment'),
    Tasks.face_quality_assessment:
    (Pipelines.face_quality_assessment,
     'damo/cv_manual_face-quality-assessment_fqa'),
    Tasks.video_multi_modal_embedding:
    (Pipelines.video_multi_modal_embedding,
     'damo/multi_modal_clip_vtretrival_msrvtt_53'),
    Tasks.text_video_retrieval: (Pipelines.prost_text_video_retrieval,
                                 'damo/multi_modal_clip_vtretrieval_prost'),
    Tasks.image_color_enhancement:
    (Pipelines.image_color_enhance,
     'damo/cv_csrnet_image-color-enhance-models'),
    Tasks.virtual_try_on: (Pipelines.virtual_try_on,
                           'damo/cv_daflow_virtual-try-on_base'),
    Tasks.image_colorization: (Pipelines.ddcolor_image_colorization,
                               'damo/cv_ddcolor_image-colorization'),
    Tasks.video_colorization: (Pipelines.video_colorization,
                               'damo/cv_unet_video-colorization'),
    Tasks.image_segmentation:
    (Pipelines.image_instance_segmentation,
     'damo/cv_swin-b_image-instance-segmentation_coco'),
    Tasks.image_driving_perception:
    (Pipelines.yolopv2_image_driving_percetion_bdd100k,
     'damo/cv_yolopv2_image-driving-perception_bdd100k'),
    Tasks.image_depth_estimation:
    (Pipelines.image_depth_estimation,
     'damo/cv_newcrfs_image-depth-estimation_indoor'),
    Tasks.image_normal_estimation:
    (Pipelines.image_normal_estimation,
     'Damo_XR_Lab/cv_omnidata_image-normal-estimation_normal'),
    Tasks.indoor_layout_estimation:
    (Pipelines.indoor_layout_estimation,
     'damo/cv_panovit_indoor-layout-estimation'),
    Tasks.video_depth_estimation:
    (Pipelines.video_depth_estimation,
     'damo/cv_dro-resnet18_video-depth-estimation_indoor'),
    Tasks.panorama_depth_estimation:
    (Pipelines.panorama_depth_estimation,
     'damo/cv_unifuse_panorama-depth-estimation'),
    Tasks.dense_optical_flow_estimation:
    (Pipelines.dense_optical_flow_estimation,
     'Damo_XR_Lab/cv_raft_dense-optical-flow_things'),
    Tasks.image_local_feature_matching:
    (Pipelines.image_local_feature_matching,
     'Damo_XR_Lab/cv_resnet-transformer_local-feature-matching_outdoor-data'),
    Tasks.image_style_transfer: (Pipelines.image_style_transfer,
                                 'damo/cv_aams_style-transfer_damo'),
    Tasks.face_image_generation: (Pipelines.face_image_generation,
                                  'damo/cv_gan_face-image-generation'),
    Tasks.image_super_resolution: (Pipelines.image_super_resolution,
                                   'damo/cv_rrdb_image-super-resolution'),
    Tasks.image_debanding: (Pipelines.image_debanding,
                            'damo/cv_rrdb_image-debanding'),
    Tasks.image_portrait_enhancement:
    (Pipelines.image_portrait_enhancement,
     'damo/cv_gpen_image-portrait-enhancement'),
    Tasks.product_retrieval_embedding:
    (Pipelines.product_retrieval_embedding,
     'damo/cv_resnet50_product-bag-embedding-models'),
    Tasks.image_to_image_generation:
    (Pipelines.image_to_image_generation,
     'damo/cv_latent_diffusion_image2image_generate'),
    Tasks.image_classification:
    (Pipelines.daily_image_classification,
     'damo/cv_vit-base_image-classification_Dailylife-labels'),
    Tasks.image_object_detection: (
        Pipelines.image_object_detection_auto,
        'damo/cv_yolox_image-object-detection-auto'),
    Tasks.ocr_recognition: (
        Pipelines.ocr_recognition,
        'damo/cv_convnextTiny_ocr-recognition-general_damo'),
    Tasks.skin_retouching: (Pipelines.skin_retouching,
                            'damo/cv_unet_skin-retouching'),
    Tasks.faq_question_answering: (
        Pipelines.faq_question_answering,
        'damo/nlp_structbert_faq-question-answering_chinese-base'),
    Tasks.crowd_counting: (Pipelines.crowd_counting,
                           'damo/cv_hrnet_crowd-counting_dcanet'),
    Tasks.video_single_object_tracking: (
        Pipelines.video_single_object_tracking,
        'damo/cv_vitb_video-single-object-tracking_ostrack'),
    Tasks.image_reid_person: (Pipelines.image_reid_person,
                              'damo/cv_passvitb_image-reid-person_market'),
    Tasks.text_driven_segmentation: (
        Pipelines.text_driven_segmentation,
        'damo/cv_vitl16_segmentation_text-driven-seg'),
    Tasks.movie_scene_segmentation: (
        Pipelines.movie_scene_segmentation,
        'damo/cv_resnet50-bert_video-scene-segmentation_movienet'),
    Tasks.shop_segmentation: (Pipelines.shop_segmentation,
                              'damo/cv_vitb16_segmentation_shop-seg'),
    Tasks.image_inpainting: (Pipelines.image_inpainting,
                             'damo/cv_fft_inpainting_lama'),
    Tasks.image_paintbyexample: (Pipelines.image_paintbyexample,
                                 'damo/cv_stable-diffusion_paint-by-example'),
    Tasks.controllable_image_generation:
    (Pipelines.controllable_image_generation,
     'dienstag/cv_controlnet_controllable-image-generation_nine-annotators'),
    Tasks.video_inpainting: (Pipelines.video_inpainting,
                             'damo/cv_video-inpainting'),
    Tasks.video_human_matting: (Pipelines.video_human_matting,
                                'damo/cv_effnetv2_video-human-matting'),
    Tasks.human_reconstruction: (Pipelines.human_reconstruction,
                                 'damo/cv_hrnet_image-human-reconstruction'),
    Tasks.text_texture_generation: (
        Pipelines.text_texture_generation,
        'damo/cv_diffuser_text-texture-generation'),
    Tasks.video_frame_interpolation: (
        Pipelines.video_frame_interpolation,
        'damo/cv_raft_video-frame-interpolation'),
    Tasks.video_deinterlace: (Pipelines.video_deinterlace,
                              'damo/cv_unet_video-deinterlace'),
    Tasks.human_wholebody_keypoint: (
        Pipelines.human_wholebody_keypoint,
        'damo/cv_hrnetw48_human-wholebody-keypoint_image'),
    Tasks.hand_static: (Pipelines.hand_static,
                        'damo/cv_mobileface_hand-static'),
    Tasks.face_human_hand_detection: (
        Pipelines.face_human_hand_detection,
        'damo/cv_nanodet_face-human-hand-detection'),
    Tasks.face_emotion: (Pipelines.face_emotion, 'damo/cv_face-emotion'),
    Tasks.product_segmentation: (Pipelines.product_segmentation,
                                 'damo/cv_F3Net_product-segmentation'),
    Tasks.referring_video_object_segmentation: (
        Pipelines.referring_video_object_segmentation,
        'damo/cv_swin-t_referring_video-object-segmentation'),
    Tasks.video_summarization: (Pipelines.video_summarization,
                                'damo/cv_googlenet_pgl-video-summarization'),
    Tasks.image_skychange: (Pipelines.image_skychange,
                            'damo/cv_hrnetocr_skychange'),
    Tasks.translation_evaluation: (
        Pipelines.translation_evaluation,
        'damo/nlp_unite_mup_translation_evaluation_multilingual_large'),
    Tasks.video_object_segmentation: (
        Pipelines.video_object_segmentation,
        'damo/cv_rdevos_video-object-segmentation'),
    Tasks.video_multi_object_tracking: (
        Pipelines.video_multi_object_tracking,
        'damo/cv_yolov5_video-multi-object-tracking_fairmot'),
    Tasks.image_multi_view_depth_estimation: (
        Pipelines.image_multi_view_depth_estimation,
        'damo/cv_casmvs_multi-view-depth-estimation_general'),
    Tasks.image_fewshot_detection: (
        Pipelines.image_fewshot_detection,
        'damo/cv_resnet101_detection_fewshot-defrcn'),
    Tasks.image_body_reshaping: (Pipelines.image_body_reshaping,
                                 'damo/cv_flow-based-body-reshaping_damo'),
    Tasks.image_face_fusion: (Pipelines.image_face_fusion,
                              'damo/cv_unet-image-face-fusion_damo'),
    Tasks.image_matching: (
        Pipelines.image_matching,
        'damo/cv_quadtree_attention_image-matching_outdoor'),
    Tasks.image_quality_assessment_mos: (
        Pipelines.image_quality_assessment_mos,
        'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC'),
    Tasks.image_quality_assessment_degradation: (
        Pipelines.image_quality_assessment_degradation,
        'damo/cv_resnet50_image-quality-assessment_degradation'),
    Tasks.vision_efficient_tuning: (
        Pipelines.vision_efficient_tuning,
        'damo/cv_vitb16_classification_vision-efficient-tuning-adapter'),
    Tasks.object_detection_3d: (Pipelines.object_detection_3d_depe,
                                'damo/cv_object-detection-3d_depe'),
    Tasks.bad_image_detecting: (Pipelines.bad_image_detecting,
                                'damo/cv_mobilenet-v2_bad-image-detecting'),
    Tasks.nerf_recon_acc: (Pipelines.nerf_recon_acc,
                           'damo/cv_nerf-3d-reconstruction-accelerate_damo'),
    Tasks.nerf_recon_4k: (Pipelines.nerf_recon_4k,
                          'damo/cv_nerf-3d-reconstruction-4k-nerf_damo'),
    Tasks.nerf_recon_vq_compression: (
        Pipelines.nerf_recon_vq_compression,
        'damo/cv_nerf-3d-reconstruction-vq-compression_damo'),
    Tasks.surface_recon_common: (Pipelines.surface_recon_common,
                                 'damo/cv_surface-reconstruction-common'),
    Tasks.siamese_uie: (Pipelines.siamese_uie,
                        'damo/nlp_structbert_siamese-uie_chinese-base'),
    Tasks.pedestrian_attribute_recognition: (
        Pipelines.pedestrian_attribute_recognition,
        'damo/cv_resnet50_pedestrian-attribute-recognition_image'),
    Tasks.text_to_360panorama_image: (
        Pipelines.text_to_360panorama_image,
        'damo/cv_diffusion_text-to-360panorama-image_generation'),
    Tasks.image_try_on: (Pipelines.image_try_on,
                         'damo/cv_SAL-VTON_virtual-try-on'),
    Tasks.human_image_generation: (Pipelines.human_image_generation,
                                   'damo/cv_FreqHPT_human-image-generation'),
    Tasks.human3d_render: (Pipelines.human3d_render,
                           'damo/cv_3d-human-synthesis-library'),
    Tasks.human3d_animation: (Pipelines.human3d_animation,
                              'damo/cv_3d-human-animation'),
    Tasks.image_view_transform: (Pipelines.image_view_transform,
                                 'damo/cv_image-view-transform'),
    Tasks.image_control_3d_portrait: (
        Pipelines.image_control_3d_portrait,
        'damo/cv_vit_image-control-3d-portrait-synthesis'),
    Tasks.self_supervised_depth_completion: (
        Pipelines.self_supervised_depth_completion,
        'damo/self-supervised-depth-completion')
}


class CVTrainers(object):
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
    ocr_recognition = 'ocr-recognition'
    ocr_detection_db = 'ocr-detection-db'
    nerf_recon_acc = 'nerf-recon-acc'
    nerf_recon_4k = 'nerf-recon-4k'
    action_detection = 'action-detection'
    vision_efficient_tuning = 'vision-efficient-tuning'
    self_supervised_depth_completion = 'self-supervised-depth-completion'


class NLPTrainers(object):
    # nlp trainers
    bert_sentiment_analysis = 'bert-sentiment-analysis'
    dialog_modeling_trainer = 'dialog-modeling-trainer'
    dialog_intent_trainer = 'dialog-intent-trainer'
    nlp_base_trainer = 'nlp-base-trainer'
    nlp_veco_trainer = 'nlp-veco-trainer'
    nlp_text_ranking_trainer = 'nlp-text-ranking-trainer'
    nlp_sentence_embedding_trainer = 'nlp-sentence-embedding-trainer'
    text_generation_trainer = 'text-generation-trainer'
    nlp_plug_trainer = 'nlp-plug-trainer'
    gpt3_trainer = 'nlp-gpt3-trainer'
    faq_question_answering_trainer = 'faq-question-answering-trainer'
    gpt_moe_trainer = 'nlp-gpt-moe-trainer'
    table_question_answering_trainer = 'table-question-answering-trainer'
    document_grounded_dialog_generate_trainer = 'document-grounded-dialog-generate-trainer'
    document_grounded_dialog_rerank_trainer = 'document-grounded-dialog-rerank-trainer'
    document_grounded_dialog_retrieval_trainer = 'document-grounded-dialog-retrieval-trainer'
    siamese_uie_trainer = 'siamese-uie-trainer'
    translation_evaluation_trainer = 'translation-evaluation-trainer'


class MultiModalTrainers(object):
    clip_multi_modal_embedding = 'clip-multi-modal-embedding'
    ofa = 'ofa'
    mplug = 'mplug'
    mgeo_ranking_trainer = 'mgeo-ranking-trainer'
    efficient_diffusion_tuning = 'efficient-diffusion-tuning'
    stable_diffusion = 'stable-diffusion'
    lora_diffusion = 'lora-diffusion'
    lora_diffusion_xl = 'lora-diffusion-xl'
    dreambooth_diffusion = 'dreambooth-diffusion'
    custom_diffusion = 'custom-diffusion'
    cones2_inference = 'cones2-inference'


class AudioTrainers(object):
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    speech_dfsmn_kws_char_farfield = 'speech_dfsmn_kws_char_farfield'
    speech_kws_fsmn_char_ctc_nearfield = 'speech_kws_fsmn_char_ctc_nearfield'
    speech_kantts_trainer = 'speech-kantts-trainer'
    speech_asr_trainer = 'speech-asr-trainer'
    speech_separation = 'speech-separation'


class Trainers(CVTrainers, NLPTrainers, MultiModalTrainers, AudioTrainers):
    """ Names for different trainer.

        Holds the standard trainer name to use for identifying different trainer.
    This should be used to register trainers.

        For a general Trainer, you can use EpochBasedTrainer.
        For a model specific Trainer, you can use ${ModelName}-${Task}-trainer.
    """

    default = 'trainer'
    tinynas_damoyolo = 'tinynas-damoyolo'

    @staticmethod
    def get_trainer_domain(attribute_or_value):
        if attribute_or_value in vars(
                CVTrainers) or attribute_or_value in vars(CVTrainers).values():
            return Fields.cv
        elif attribute_or_value in vars(
                NLPTrainers) or attribute_or_value in vars(
                    NLPTrainers).values():
            return Fields.nlp
        elif attribute_or_value in vars(
                AudioTrainers) or attribute_or_value in vars(
                    AudioTrainers).values():
            return Fields.audio
        elif attribute_or_value in vars(
                MultiModalTrainers) or attribute_or_value in vars(
                    MultiModalTrainers).values():
            return Fields.multi_modal
        elif attribute_or_value == Trainers.default:
            return Trainers.default
        else:
            return 'unknown'


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
    image_driving_perception_preprocessor = 'image-driving-perception-preprocessor'
    image_portrait_enhancement_preprocessor = 'image-portrait-enhancement-preprocessor'
    image_quality_assessment_man_preprocessor = 'image-quality_assessment-man-preprocessor'
    image_quality_assessment_mos_preprocessor = 'image-quality_assessment-mos-preprocessor'
    video_summarization_preprocessor = 'video-summarization-preprocessor'
    movie_scene_segmentation_preprocessor = 'movie-scene-segmentation-preprocessor'
    image_classification_bypass_preprocessor = 'image-classification-bypass-preprocessor'
    object_detection_scrfd = 'object-detection-scrfd'
    image_sky_change_preprocessor = 'image-sky-change-preprocessor'
    image_demoire_preprocessor = 'image-demoire-preprocessor'
    ocr_recognition = 'ocr-recognition'
    ocr_detection = 'ocr-detection'
    bad_image_detecting_preprocessor = 'bad-image-detecting-preprocessor'
    nerf_recon_acc_preprocessor = 'nerf-recon-acc-preprocessor'
    nerf_recon_4k_preprocessor = 'nerf-recon-4k-preprocessor'
    nerf_recon_vq_compression_preprocessor = 'nerf-recon-vq-compression-preprocessor'
    controllable_image_generation_preprocessor = 'controllable-image-generation-preprocessor'
    image_classification_preprocessor = 'image-classification-preprocessor'

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
    word_alignment = 'word-alignment'
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
    canmt_translation = 'canmt-translation'
    dialog_use_preprocessor = 'dialog-use-preprocessor'
    siamese_uie_preprocessor = 'siamese-uie-preprocessor'
    document_grounded_dialog_retrieval = 'document-grounded-dialog-retrieval'
    document_grounded_dialog_rerank = 'document-grounded-dialog-rerank'
    document_grounded_dialog_generate = 'document-grounded-dialog-generate'
    machine_reading_comprehension_for_ner = 'machine-reading-comprehension-for-ner'

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
    diffusion_image_generation_preprocessor = 'diffusion-image-generation-preprocessor'
    mplug_owl_preprocessor = 'mplug-owl-preprocessor'
    image_captioning_clip_interrogator_preprocessor = 'image-captioning-clip-interrogator-preprocessor'

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
    # loss metric
    loss_metric = 'loss-metric'
    # metrics for token-classification task
    token_cls_metric = 'token-cls-metric'
    # metrics for text-generation task
    text_gen_metric = 'text-gen-metric'
    # file saving wrapper
    prediction_saving_wrapper = 'prediction-saving-wrapper'
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
    # metirc for image-quality-assessment-mos task
    image_quality_assessment_mos_metric = 'image-quality-assessment-mos-metric'
    # metirc for image-quality-assessment-degradation task
    image_quality_assessment_degradation_metric = 'image-quality-assessment-degradation-metric'
    # metric for text-ranking task
    text_ranking_metric = 'text-ranking-metric'
    # metric for image-colorization task
    image_colorization_metric = 'image-colorization-metric'
    ocr_recognition_metric = 'ocr-recognition-metric'
    # metric for translation evaluation
    translation_evaluation_metric = 'translation-evaluation-metric'


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
    LoadCheckpointHook = 'LoadCheckpointHook'

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
    MegatronHook = 'MegatronHook'
    DDPHook = 'DDPHook'
    SwiftHook = 'SwiftHook'


class LR_Schedulers(object):
    """learning rate scheduler is defined here

    """
    LinearWarmup = 'LinearWarmup'
    ConstantWarmup = 'ConstantWarmup'
    ExponentialWarmup = 'ExponentialWarmup'


class CustomDatasets(object):
    """ Names for different datasets.
    """
    PairedDataset = 'PairedDataset'
    SiddDataset = 'SiddDataset'
    GoproDataset = 'GoproDataset'
    RedsDataset = 'RedsDataset'
