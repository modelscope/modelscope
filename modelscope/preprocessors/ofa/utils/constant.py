# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.constant import Tasks

OFA_TASK_KEY_MAPPING = {
    Tasks.ocr_recognition: ['image'],
    Tasks.image_captioning: ['image'],
    Tasks.image_classification: ['image'],
    Tasks.text_summarization: ['text'],
    Tasks.text_classification: ['text', 'text2'],
    Tasks.visual_grounding: ['image', 'text'],
    Tasks.visual_question_answering: ['image', 'text'],
    Tasks.visual_entailment: ['image', 'text', 'text2'],
    Tasks.text_to_image_synthesis: ['text'],
    Tasks.auto_speech_recognition: ['wav', 'text'],
    Tasks.sudoku: ['text'],
    Tasks.text2sql: ['text', 'database'],
}
