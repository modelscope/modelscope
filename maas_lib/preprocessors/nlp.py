# Copyright (c) Alibaba, Inc. and its affiliates.

import uuid
from typing import Any, Dict, Union

from transformers import AutoTokenizer

from maas_lib.utils.constant import Fields, InputFields
from maas_lib.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = [
    'Tokenize',
    'SequenceClassificationPreprocessor',
]


@PREPROCESSORS.register_module(Fields.nlp)
class Tokenize(Preprocessor):

    def __init__(self, tokenizer_name) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(data, str):
            data = {InputFields.text: data}
        token_dict = self._tokenizer(data[InputFields.text])
        data.update(token_dict)
        return data


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=r'bert-sentiment-analysis')
class SequenceClassificationPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        from easynlp.modelzoo import AutoTokenizer
        self.model_dir: str = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.sequence_length = kwargs.pop('sequence_length', 128)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    @type_assert(object, str)
    def __call__(self, data: str) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        new_data = {self.first_sequence: data}
        # preprocess the data for the model input

        rst = {
            'id': [],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }

        max_seq_length = self.sequence_length

        text_a = new_data[self.first_sequence]
        text_b = new_data.get(self.second_sequence, None)
        feature = self.tokenizer(
            text_a,
            text_b,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length)

        rst['id'].append(new_data.get('id', str(uuid.uuid4())))
        rst['input_ids'].append(feature['input_ids'])
        rst['attention_mask'].append(feature['attention_mask'])
        rst['token_type_ids'].append(feature['token_type_ids'])

        return rst
