# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

OFA_TASK_KEY_MAPPING = {
    Tasks.ocr_recognition: OutputKeys.TEXT,
    Tasks.image_captioning: OutputKeys.CAPTION,
    Tasks.text_summarization: OutputKeys.TEXT,
    Tasks.visual_question_answering: OutputKeys.TEXT,
    Tasks.visual_grounding: OutputKeys.BOXES,
    Tasks.text_classification: OutputKeys.LABELS,
    Tasks.image_classification: OutputKeys.LABELS,
    Tasks.visual_entailment: OutputKeys.LABELS,
    Tasks.auto_speech_recognition: OutputKeys.TEXT,
    Tasks.sudoku: OutputKeys.TEXT,
    Tasks.text2sql: OutputKeys.TEXT,
}
