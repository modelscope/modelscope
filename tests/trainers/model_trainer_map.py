model_trainer_map = {
    'damo/speech_frcrn_ans_cirm_16k':
    ['tests/trainers/audio/test_ans_trainer.py'],
    'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch':
    ['tests/trainers/audio/test_asr_trainer.py'],
    'damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya':
    ['tests/trainers/audio/test_kws_farfield_trainer.py'],
    'damo/speech_charctc_kws_phone-xiaoyun':
    ['tests/trainers/audio/test_kws_nearfield_trainer.py'],
    'damo/speech_mossformer_separation_temporal_8k':
    ['tests/trainers/audio/test_separation_trainer.py'],
    'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_16k':
    ['tests/trainers/audio/test_tts_trainer.py'],
    'damo/cv_resnet_carddetection_scrfd34gkps':
    ['tests/trainers/test_card_detection_scrfd_trainer.py'],
    'damo/multi-modal_clip-vit-base-patch16_zh':
    ['tests/trainers/test_clip_trainer.py'],
    'damo/nlp_space_pretrained-dialog-model':
    ['tests/trainers/test_dialog_intent_trainer.py'],
    'damo/cv_resnet_facedetection_scrfd10gkps':
    ['tests/trainers/test_face_detection_scrfd_trainer.py'],
    'damo/nlp_structbert_faq-question-answering_chinese-base':
    ['tests/trainers/test_finetune_faq_question_answering.py'],
    'PAI/nlp_gpt3_text-generation_0.35B_MoE-64':
    ['tests/trainers/test_finetune_gpt_moe.py'],
    'damo/nlp_gpt3_text-generation_1.3B': [
        'tests/trainers/test_finetune_gpt3.py'
    ],
    'damo/mgeo_backbone_chinese_base': [
        'tests/trainers/test_finetune_mgeo.py'
    ],
    'damo/mplug_backbone_base_en': ['tests/trainers/test_finetune_mplug.py'],
    'damo/nlp_structbert_backbone_base_std': [
        'tests/trainers/test_finetune_sequence_classification.py',
        'tests/trainers/test_finetune_token_classification.py'
    ],
    'damo/nlp_palm2.0_text-generation_english-base': [
        'tests/trainers/test_finetune_text_generation.py'
    ],
    'damo/nlp_gpt3_text-generation_chinese-base': [
        'tests/trainers/test_finetune_text_generation.py'
    ],
    'damo/nlp_palm2.0_text-generation_chinese-base': [
        'tests/trainers/test_finetune_text_generation.py'
    ],
    'damo/nlp_corom_passage-ranking_english-base': [
        'tests/trainers/test_finetune_text_ranking.py'
    ],
    'damo/nlp_rom_passage-ranking_chinese-base': [
        'tests/trainers/test_finetune_text_ranking.py'
    ],
    'damo/cv_nextvit-small_image-classification_Dailylife-labels': [
        'tests/trainers/test_general_image_classification_trainer.py'
    ],
    'damo/cv_convnext-base_image-classification_garbage': [
        'tests/trainers/test_general_image_classification_trainer.py'
    ],
    'damo/cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k': [
        'tests/trainers/test_general_image_classification_trainer.py'
    ],
    'damo/cv_csrnet_image-color-enhance-models': [
        'tests/trainers/test_image_color_enhance_trainer.py'
    ],
    'damo/cv_nafnet_image-deblur_gopro': [
        'tests/trainers/test_image_deblur_trainer.py'
    ],
    'damo/cv_resnet101_detection_fewshot-defrcn': [
        'tests/trainers/test_image_defrcn_fewshot_trainer.py'
    ],
    'damo/cv_nafnet_image-denoise_sidd': [
        'tests/trainers/test_image_denoise_trainer.py'
    ],
    'damo/cv_fft_inpainting_lama': [
        'tests/trainers/test_image_inpainting_trainer.py'
    ],
    'damo/cv_swin-b_image-instance-segmentation_coco': [
        'tests/trainers/test_image_instance_segmentation_trainer.py'
    ],
    'damo/cv_gpen_image-portrait-enhancement': [
        'tests/trainers/test_image_portrait_enhancement_trainer.py'
    ],
    'damo/cv_clip-it_video-summarization_language-guided_en': [
        'tests/trainers/test_language_guided_video_summarization_trainer.py'
    ],
    'damo/cv_resnet50-bert_video-scene-segmentation_movienet': [
        'tests/trainers/test_movie_scene_segmentation_trainer.py'
    ],
    'damo/ofa_mmspeech_pretrain_base_zh': [
        'tests/trainers/test_ofa_mmspeech_trainer.py'
    ],
    'damo/ofa_ocr-recognition_scene_base_zh': [
        'tests/trainers/test_ofa_trainer.py'
    ],
    'damo/nlp_plug_text-generation_27B': [
        'tests/trainers/test_plug_finetune_text_generation.py'
    ],
    'damo/cv_swin-t_referring_video-object-segmentation': [
        'tests/trainers/test_referring_video_object_segmentation_trainer.py'
    ],
    'damo/nlp_convai_text2sql_pretrain_cn': [
        'tests/trainers/test_table_question_answering_trainer.py'
    ],
    'damo/multi-modal_team-vit-large-patch14_multi-modal-similarity': [
        'tests/trainers/test_team_transfer_trainer.py'
    ],
    'damo/cv_tinynas_object-detection_damoyolo': [
        'tests/trainers/test_tinynas_damoyolo_trainer.py'
    ],
    'damo/nlp_structbert_sentence-similarity_chinese-tiny': [
        'tests/trainers/test_trainer_with_nlp.py'
    ],
    'damo/nlp_structbert_sentiment-classification_chinese-base': [
        'tests/trainers/test_trainer_with_nlp.py'
    ],
    'damo/nlp_structbert_sentence-similarity_chinese-base': [
        'tests/trainers/test_trainer_with_nlp.py'
    ],
    'damo/nlp_csanmt_translation_en2zh': [
        'tests/trainers/test_translation_trainer.py'
    ],
    'damo/nlp_csanmt_translation_en2fr': [
        'tests/trainers/test_translation_trainer.py'
    ],
    'damo/nlp_csanmt_translation_en2es': [
        'tests/trainers/test_translation_trainer.py'
    ],
    'damo/nlp_unite_mup_translation_evaluation_multilingual_base': [
        'tests/trainers/test_translation_evaluation_trainer.py'
    ],
    'damo/nlp_unite_mup_translation_evaluation_multilingual_large': [
        'tests/trainers/test_translation_evaluation_trainer.py'
    ],
    'damo/cv_googlenet_pgl-video-summarization': [
        'tests/trainers/test_video_summarization_trainer.py'
    ],
}
