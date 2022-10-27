# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Tuple, Union

import torch

from modelscope.metainfo import Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert
from .token_classification_preprocessor import TokenClassificationPreprocessor


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.viet_ner_tokenizer)
class NERPreprocessorViet(TokenClassificationPreprocessor):

    @type_assert(object, str)
    def __call__(self, data: str) -> Dict[str, Any]:
        from pyvi import ViTokenizer

        seg_words = [
            t.strip(' ') for t in ViTokenizer.tokenize(data).split(' ')
            if t.strip(' ') != ''
        ]
        raw_words = []
        for w in seg_words:
            raw_words.extend(w.split('_'))
        segmented_data = ' '.join(raw_words)
        output = super().__call__(segmented_data)

        return output
