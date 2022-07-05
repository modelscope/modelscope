# Copyright (c) Alibaba, Inc. and its affiliates.


class Models(object):
    """ Names for different models.

        Holds the standard model name to use for identifying different model.
    This should be used to register models.

        Model name should only contain model info but not task info.
    """
    # vision models

    # nlp models
    bert = 'bert'
    palm = 'palm-v2'
    structbert = 'structbert'
    veco = 'veco'
    translation = 'csanmt-translation'
    space = 'space'

    # audio models
    sambert_hifi_16k = 'sambert-hifi-16k'
    generic_tts_frontend = 'generic-tts-frontend'
    hifigan16k = 'hifigan16k'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    kws_kwsbp = 'kws-kwsbp'

    # multi-modal models
    ofa = 'ofa'
    clip = 'clip-multi-modal-embedding'
    mplug = 'mplug'
    imagen = 'imagen-text-to-image-synthesis'


class Pipelines(object):
    """ Names for different pipelines.

        Holds the standard pipline name to use for identifying different pipeline.
    This should be used to register pipelines.

        For pipeline which support different models and implements the common function, we
    should use task name for this pipeline.
        For pipeline which suuport only one model, we should use ${Model}-${Task} as its name.
    """
    # vision tasks
    image_matting = 'unet-image-matting'
    person_image_cartoon = 'unet-person-image-cartoon'
    ocr_detection = 'resnet18-ocr-detection'
    action_recognition = 'TAdaConv_action-recognition'
    animal_recognation = 'resnet101-animal_recog'
    cmdssl_video_embedding = 'cmdssl-r2p1d_video_embedding'

    # nlp tasks
    sentence_similarity = 'sentence-similarity'
    word_segmentation = 'word-segmentation'
    text_generation = 'text-generation'
    sentiment_analysis = 'sentiment-analysis'
    sentiment_classification = 'sentiment-classification'
    fill_mask = 'fill-mask'
    csanmt_translation = 'csanmt-translation'
    nli = 'nli'
    dialog_intent_prediction = 'dialog-intent-prediction'
    dialog_modeling = 'dialog-modeling'
    dialog_state_tracking = 'dialog-state-tracking'
    zero_shot_classification = 'zero-shot-classification'

    # audio tasks
    sambert_hifigan_16k_tts = 'sambert-hifigan-16k-tts'
    speech_dfsmn_aec_psm_16k = 'speech-dfsmn-aec-psm-16k'
    speech_frcrn_ans_cirm_16k = 'speech_frcrn_ans_cirm_16k'
    kws_kwsbp = 'kws-kwsbp'

    # multi-modal tasks
    image_caption = 'image-captioning'
    multi_modal_embedding = 'multi-modal-embedding'
    visual_question_answering = 'visual-question-answering'
    text_to_image_synthesis = 'text-to-image-synthesis'


class Trainers(object):
    """ Names for different trainer.

        Holds the standard trainer name to use for identifying different trainer.
    This should be used to register trainers.

        For a general Trainer, you can use easynlp-trainer/ofa-trainer/sofa-trainer.
        For a model specific Trainer, you can use ${ModelName}-${Task}-trainer.
    """

    default = 'Trainer'


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

    # nlp preprocessor
    sen_sim_tokenizer = 'sen-sim-tokenizer'
    bert_seq_cls_tokenizer = 'bert-seq-cls-tokenizer'
    palm_text_gen_tokenizer = 'palm-text-gen-tokenizer'
    token_cls_tokenizer = 'token-cls-tokenizer'
    nli_tokenizer = 'nli-tokenizer'
    sen_cls_tokenizer = 'sen-cls-tokenizer'
    dialog_intent_preprocessor = 'dialog-intent-preprocessor'
    dialog_modeling_preprocessor = 'dialog-modeling-preprocessor'
    dialog_state_tracking_preprocessor = 'dialog-state-tracking-preprocessor'
    sbert_token_cls_tokenizer = 'sbert-token-cls-tokenizer'
    zero_shot_cls_tokenizer = 'zero-shot-cls-tokenizer'

    # audio preprocessor
    linear_aec_fbank = 'linear-aec-fbank'
    text_to_tacotron_symbols = 'text-to-tacotron-symbols'
    wav_to_lists = 'wav-to-lists'

    # multi-modal
    ofa_image_caption = 'ofa-image-caption'
    mplug_visual_question_answering = 'mplug-visual-question-answering'
