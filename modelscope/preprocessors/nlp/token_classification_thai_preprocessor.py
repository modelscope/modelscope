# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert
from .token_classification_preprocessor import \
    TokenClassificationTransformersPreprocessor


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.thai_ner_tokenizer)
class NERPreprocessorThai(TokenClassificationTransformersPreprocessor):

    @type_assert(object, (str, dict))
    def __call__(self, data: Union[Dict, str]) -> Dict[str, Any]:
        from pythainlp import word_tokenize
        if isinstance(data, str):
            text = data
        else:
            text = data[self.first_sequence]
        segmented_data = ' '.join([
            w.strip(' ') for w in word_tokenize(text=text, engine='newmm')
            if w.strip(' ') != ''
        ])
        output = super().__call__(segmented_data)

        return output


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.thai_wseg_tokenizer)
class WordSegmentationPreprocessorThai(
        TokenClassificationTransformersPreprocessor):

    @type_assert(object, (str, dict))
    def __call__(self, data: Union[Dict, str]) -> Dict[str, Any]:
        import regex
        if isinstance(data, str):
            text = data
        else:
            text = data[self.first_sequence]
        data = regex.findall(r'\X', text)
        data = ' '.join([char for char in data])

        output = super().__call__(data)

        return output
