# Copyright (c) Alibaba, Inc. and its affiliates.

import uuid
from typing import Any, Dict, Union

from transformers import AutoTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.utils.constant import Fields, InputFields
from modelscope.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = [
    'Tokenize', 'SequenceClassificationPreprocessor',
    'TextGenerationPreprocessor', 'TokenClassifcationPreprocessor',
    'FillMaskPreprocessor'
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
    Fields.nlp, module_name=Preprocessors.bert_seq_cls_tokenizer)
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
        print(f'this is the tokenzier {self.tokenizer}')

    @type_assert(object, (str, tuple, Dict))
    def __call__(self, data: Union[str, tuple, Dict]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str or tuple, Dict):
            sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
            or
            (sentence1, sentence2)
                sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
                sentence2 (str): a sentence
                    Example:
                        'you are so beautiful.'
            or
            {field1: field_value1, field2: field_value2}
            field1 (str): field name, default 'first_sequence'
            field_value1 (str): a sentence
                    Example:
                        'you are so handsome.'

            field2 (str): field name, default 'second_sequence'
            field_value2 (str): a sentence
                Example:
                    'you are so beautiful.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if isinstance(data, str):
            new_data = {self.first_sequence: data}
        elif isinstance(data, tuple):
            sentence1, sentence2 = data
            new_data = {
                self.first_sequence: sentence1,
                self.second_sequence: sentence2
            }
        else:
            new_data = data

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


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.palm_text_gen_tokenizer)
class TextGenerationPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, tokenizer, *args, **kwargs):
        """preprocess the data using the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
        self.second_sequence: str = kwargs.pop('second_sequence',
                                               'second_sequence')
        self.sequence_length: int = kwargs.pop('sequence_length', 128)
        self.tokenizer = tokenizer

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
        import torch

        new_data = {self.first_sequence: data}
        # preprocess the data for the model input

        rst = {'input_ids': [], 'attention_mask': []}

        max_seq_length = self.sequence_length

        text_a = new_data.get(self.first_sequence, None)
        text_b = new_data.get(self.second_sequence, None)
        feature = self.tokenizer(
            text_a,
            text_b,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length)

        rst['input_ids'].append(feature['input_ids'])
        rst['attention_mask'].append(feature['attention_mask'])

        return {k: torch.tensor(v) for k, v in rst.items()}


@PREPROCESSORS.register_module(Fields.nlp)
class FillMaskPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        from sofa.utils.backend import AutoTokenizer
        self.model_dir = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
        self.sequence_length = kwargs.pop('sequence_length', 128)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=False)

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
        import torch

        new_data = {self.first_sequence: data}
        # preprocess the data for the model input

        rst = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}

        max_seq_length = self.sequence_length

        text_a = new_data[self.first_sequence]
        feature = self.tokenizer(
            text_a,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_token_type_ids=True)

        rst['input_ids'].append(feature['input_ids'])
        rst['attention_mask'].append(feature['attention_mask'])
        rst['token_type_ids'].append(feature['token_type_ids'])

        return {k: torch.tensor(v) for k, v in rst.items()}


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sbert_token_cls_tokenizer)
class TokenClassifcationPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        from sofa import SbertTokenizer
        self.model_dir: str = model_dir
        self.tokenizer = SbertTokenizer.from_pretrained(self.model_dir)

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
        # preprocess the data for the model input

        text = data.replace(' ', '').strip()
        tokens = []
        for token in text:
            token = self.tokenizer.tokenize(token)
            tokens.extend(token)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
