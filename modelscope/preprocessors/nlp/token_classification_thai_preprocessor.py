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
    Fields.nlp, module_name=Preprocessors.thai_ner_tokenizer)
class NERPreprocessorThai(TokenClassificationPreprocessor):

    @type_assert(object, str)
    def __call__(self, data: str) -> Dict[str, Any]:
        from pythainlp import word_tokenize

        segmented_data = ' '.join([
            w.strip(' ') for w in word_tokenize(text=data, engine='newmm')
            if w.strip(' ') != ''
        ])
        output = super().__call__(segmented_data)

        return output


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.thai_wseg_tokenizer)
class WordSegmentationPreprocessorThai(TokenClassificationPreprocessor):

    @type_assert(object, str)
    def __call__(self, data: str) -> Dict[str, Any]:
        import regex
        data = regex.findall(r'\X', data)
        data = ' '.join([char for char in data])

        output = super().__call__(data)

        return output
