# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from PIL import Image

from modelscope.utils.constant import Tasks


class InputKeys(object):
    IMAGE = 'image'
    TEXT = 'text'
    VIDEO = 'video'


class InputType(object):
    IMAGE = 'image'
    TEXT = 'text'
    AUDIO = 'audio'
    VIDEO = 'video'
    BOX = 'box'
    DICT = 'dict'
    LIST = 'list'
    INT = 'int'


INPUT_TYPE = {
    InputType.IMAGE: (str, np.ndarray, Image.Image),
    InputType.TEXT: str,
    InputType.AUDIO: (str, bytes, np.ndarray),
    InputType.VIDEO: (str, np.ndarray, 'cv2.VideoCapture'),
    InputType.BOX: (list, np.ndarray),
    InputType.DICT: (dict, type(None)),
    InputType.LIST: (list, type(None)),
    InputType.INT: int,
}


def check_input_type(input_type, input):
    expected_type = INPUT_TYPE[input_type]
    if input_type == InputType.VIDEO:
        # special type checking using class name, to avoid introduction of opencv dependency into fundamental framework.
        assert type(input).__name__ == 'VideoCapture' or isinstance(input, expected_type),\
            f'invalid input type for {input_type}, expected {expected_type} but got {type(input)}\n {input}'
    else:
        assert isinstance(input, expected_type), \
            f'invalid input type for {input_type}, expected {expected_type} but got {type(input)}\n {input}'


TASK_INPUTS = {
    # if task input is single var, value is  InputType
    # if task input is a tuple,  value is tuple of InputType
    # if task input is a dict, value is a dict of InputType, where key
    # equals the one needed in pipeline input dict
    # if task input is a list, value is a set of input format, in which
    # each element corresponds to one input format as described above.
    # ============ vision tasks ===================
    Tasks.ocr_detection:
    InputType.IMAGE,
    Tasks.ocr_recognition:
    InputType.IMAGE,
    Tasks.face_2d_keypoints:
    InputType.IMAGE,
    Tasks.face_detection:
    InputType.IMAGE,
    Tasks.facial_expression_recognition:
    InputType.IMAGE,
    Tasks.face_attribute_recognition:
    InputType.IMAGE,
    Tasks.face_recognition:
    InputType.IMAGE,
    Tasks.face_reconstruction:
    InputType.IMAGE,
    Tasks.human_detection:
    InputType.IMAGE,
    Tasks.face_image_generation:
    InputType.INT,
    Tasks.image_classification:
    InputType.IMAGE,
    Tasks.image_object_detection:
    InputType.IMAGE,
    Tasks.domain_specific_object_detection:
    InputType.IMAGE,
    Tasks.image_segmentation:
    InputType.IMAGE,
    Tasks.portrait_matting:
    InputType.IMAGE,
    Tasks.image_fewshot_detection:
    InputType.IMAGE,
    Tasks.open_vocabulary_detection: {
        'img': InputType.IMAGE,
        'category_names': InputType.TEXT
    },
    Tasks.image_driving_perception:
    InputType.IMAGE,
    Tasks.vision_efficient_tuning:
    InputType.IMAGE,

    # image editing task result for a single image
    Tasks.skin_retouching:
    InputType.IMAGE,
    Tasks.image_super_resolution:
    InputType.IMAGE,
    Tasks.image_colorization:
    InputType.IMAGE,
    Tasks.image_color_enhancement:
    InputType.IMAGE,
    Tasks.image_denoising:
    InputType.IMAGE,
    Tasks.image_portrait_enhancement:
    InputType.IMAGE,
    Tasks.crowd_counting:
    InputType.IMAGE,
    Tasks.image_inpainting: {
        'img': InputType.IMAGE,
        'mask': InputType.IMAGE,
    },
    Tasks.image_paintbyexample: {
        'img': InputType.IMAGE,
        'mask': InputType.IMAGE,
        'reference': InputType.IMAGE,
    },
    Tasks.image_skychange: {
        'sky_image': InputType.IMAGE,
        'scene_image': InputType.IMAGE,
    },
    Tasks.controllable_image_generation: {
        'image': InputType.IMAGE,
        'prompt': InputType.TEXT,
    },
    Tasks.video_colorization:
    InputType.VIDEO,

    # image generation task result for a single image
    Tasks.image_to_image_generation:
    InputType.IMAGE,
    Tasks.image_to_image_translation:
    InputType.IMAGE,
    Tasks.image_style_transfer: {
        'content': InputType.IMAGE,
        'style': InputType.IMAGE,
    },
    Tasks.image_portrait_stylization:
    InputType.IMAGE,
    Tasks.live_category:
    InputType.VIDEO,
    Tasks.action_recognition:
    InputType.VIDEO,
    Tasks.body_2d_keypoints:
    InputType.IMAGE,
    Tasks.body_3d_keypoints:
    InputType.VIDEO,
    Tasks.hand_2d_keypoints:
    InputType.IMAGE,
    Tasks.video_single_object_tracking: (InputType.VIDEO, InputType.BOX),
    Tasks.video_multi_object_tracking:
    InputType.VIDEO,
    Tasks.video_category:
    InputType.VIDEO,
    Tasks.product_retrieval_embedding:
    InputType.IMAGE,
    Tasks.video_embedding:
    InputType.VIDEO,
    Tasks.virtual_try_on: (InputType.IMAGE, InputType.IMAGE, InputType.IMAGE),
    Tasks.text_driven_segmentation: {
        InputKeys.IMAGE: InputType.IMAGE,
        InputKeys.TEXT: InputType.TEXT
    },
    Tasks.shop_segmentation:
    InputType.IMAGE,
    Tasks.movie_scene_segmentation:
    InputType.VIDEO,
    Tasks.bad_image_detecting:
    InputType.IMAGE,

    # ============ nlp tasks ===================
    Tasks.text_classification: [
        InputType.TEXT,
        (InputType.TEXT, InputType.TEXT),
        {
            'text': InputType.TEXT,
            'text2': InputType.TEXT
        },
    ],
    Tasks.sentence_similarity: (InputType.TEXT, InputType.TEXT),
    Tasks.nli: (InputType.TEXT, InputType.TEXT),
    Tasks.sentiment_classification:
    InputType.TEXT,
    Tasks.zero_shot_classification:
    InputType.TEXT,
    Tasks.relation_extraction:
    InputType.TEXT,
    Tasks.translation:
    InputType.TEXT,
    Tasks.word_segmentation: [InputType.TEXT, {
        'text': InputType.TEXT,
    }],
    Tasks.part_of_speech:
    InputType.TEXT,
    Tasks.named_entity_recognition:
    InputType.TEXT,
    Tasks.text_error_correction:
    InputType.TEXT,
    Tasks.sentence_embedding: {
        'source_sentence': InputType.LIST,
        'sentences_to_compare': InputType.LIST,
    },
    Tasks.text_ranking: (InputType.TEXT, InputType.TEXT),
    Tasks.text_generation:
    InputType.TEXT,
    Tasks.fid_dialogue: {
        'history': InputType.TEXT,
        'knowledge': InputType.TEXT,
        'bot_profile': InputType.TEXT,
        'user_profile': InputType.TEXT,
    },
    Tasks.fill_mask:
    InputType.TEXT,
    Tasks.task_oriented_conversation: {
        'user_input': InputType.TEXT,
        'history': InputType.DICT,
    },
    Tasks.table_question_answering: {
        'question': InputType.TEXT,
        'history_sql': InputType.DICT,
    },
    Tasks.faq_question_answering: {
        'query_set': InputType.LIST,
        'support_set': InputType.LIST,
    },
    Tasks.translation_evaluation: {
        'hyp': InputType.LIST,
        'src': InputType.LIST,
        'ref': InputType.LIST,
    },
    Tasks.sudoku:
    InputType.TEXT,
    Tasks.text2sql: {
        'text': InputType.TEXT,
        'database': InputType.TEXT
    },
    Tasks.document_grounded_dialog_generate: {
        'query': InputType.LIST,
        'context': InputType.LIST,
        'label': InputType.LIST,
    },
    Tasks.document_grounded_dialog_rerank: {
        'dataset': InputType.LIST
    },
    Tasks.document_grounded_dialog_retrieval: {
        'query': InputType.LIST,
        'positive': InputType.LIST,
        'negative': InputType.LIST
    },

    # ============ audio tasks ===================
    Tasks.auto_speech_recognition:
    [InputType.AUDIO, {
        'wav': InputType.AUDIO,
        'text': InputType.TEXT
    }],
    Tasks.speech_signal_process:
    InputType.AUDIO,
    Tasks.acoustic_echo_cancellation: {
        'nearend_mic': InputType.AUDIO,
        'farend_speech': InputType.AUDIO
    },
    Tasks.speech_separation:
    InputType.AUDIO,
    Tasks.acoustic_noise_suppression:
    InputType.AUDIO,
    Tasks.text_to_speech:
    InputType.TEXT,
    Tasks.keyword_spotting:
    InputType.AUDIO,
    Tasks.inverse_text_processing:
    InputType.TEXT,

    # ============ multi-modal tasks ===================
    Tasks.image_captioning: [InputType.IMAGE, {
        'image': InputType.IMAGE,
    }],
    Tasks.video_captioning: [InputType.VIDEO, {
        'video': InputType.VIDEO,
    }],
    Tasks.visual_grounding: {
        'image': InputType.IMAGE,
        'text': InputType.TEXT
    },
    Tasks.text_to_image_synthesis: {
        'text': InputType.TEXT,
    },
    Tasks.multi_modal_embedding: {
        'img': InputType.IMAGE,
        'text': InputType.TEXT
    },
    Tasks.generative_multi_modal_embedding: {
        'image': InputType.IMAGE,
        'text': InputType.TEXT
    },
    Tasks.multi_modal_similarity: {
        'img': InputType.IMAGE,
        'text': InputType.TEXT
    },
    Tasks.visual_question_answering: {
        'image': InputType.IMAGE,
        'text': InputType.TEXT
    },
    Tasks.video_question_answering: {
        'video': InputType.VIDEO,
        'text': InputType.TEXT
    },
    Tasks.visual_entailment: {
        'image': InputType.IMAGE,
        'text': InputType.TEXT,
        'text2': InputType.TEXT,
    },
    Tasks.action_detection:
    InputType.VIDEO,
    Tasks.image_reid_person:
    InputType.IMAGE,
    Tasks.video_inpainting: {
        'video_input_path': InputType.TEXT,
        'video_output_path': InputType.TEXT,
        'mask_path': InputType.TEXT,
    }
}
