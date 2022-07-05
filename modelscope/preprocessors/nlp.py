# Copyright (c) Alibaba, Inc. and its affiliates.

import uuid
from typing import Any, Dict, Union

from transformers import AutoTokenizer

from ..metainfo import Preprocessors
from ..models import Model
from ..utils.constant import Fields, InputFields
from ..utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = [
    'Tokenize', 'SequenceClassificationPreprocessor',
    'TextGenerationPreprocessor', 'TokenClassificationPreprocessor',
    'NLIPreprocessor', 'SentimentClassificationPreprocessor',
    'FillMaskPreprocessor', 'SentenceSimilarityPreprocessor',
    'ZeroShotClassificationPreprocessor'
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


class NLPPreprocessorBase(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)
        self.model_dir: str = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.tokenize_kwargs = kwargs
        self.tokenizer = self.build_tokenizer(model_dir)

    def build_tokenizer(self, model_dir):
        from sofa import SbertTokenizer
        return SbertTokenizer.from_pretrained(model_dir)

    @type_assert(object, object)
    def __call__(self, data: Union[str, tuple, Dict]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (tuple): [sentence1, sentence2]
                sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
                sentence2 (str): a sentence
                    Example:
                        'you are so beautiful.'
        Returns:
            Dict[str, Any]: the preprocessed data
        """

        text_a, text_b = None, None
        if isinstance(data, str):
            text_a = data
        elif isinstance(data, tuple):
            assert len(data) == 2
            text_a, text_b = data
        elif isinstance(data, dict):
            text_a = data.get(self.first_sequence)
            text_b = data.get(self.second_sequence, None)

        return self.tokenizer(text_a, text_b, **self.tokenize_kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.nli_tokenizer)
class NLIPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        kwargs['truncation'] = True
        kwargs['padding'] = False
        kwargs['return_tensors'] = 'pt'
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, *args, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_cls_tokenizer)
class SentimentClassificationPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        kwargs['truncation'] = True
        kwargs['padding'] = 'max_length'
        kwargs['return_tensors'] = 'pt'
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, *args, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_sim_tokenizer)
class SentenceSimilarityPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        kwargs['truncation'] = True
        kwargs['padding'] = False
        kwargs['return_tensors'] = 'pt'
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, *args, **kwargs)


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
        feature = super().__call__(data)
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
class TextGenerationPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        kwargs['truncation'] = True
        kwargs['padding'] = 'max_length'
        kwargs['return_tensors'] = 'pt'
        kwargs['return_token_type_ids'] = False
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, *args, **kwargs)

    def build_tokenizer(self, model_dir):
        return self.tokenizer


@PREPROCESSORS.register_module(Fields.nlp)
class FillMaskPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        kwargs['truncation'] = True
        kwargs['padding'] = 'max_length'
        kwargs['return_tensors'] = 'pt'
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        kwargs['return_token_type_ids'] = True
        super().__init__(model_dir, *args, **kwargs)

    def build_tokenizer(self, model_dir):
        from ..utils.hub import get_model_type
        model_type = get_model_type(model_dir)
        if model_type in ['sbert', 'structbert', 'bert']:
            from sofa import SbertTokenizer
            return SbertTokenizer.from_pretrained(model_dir, use_fast=False)
        elif model_type == 'veco':
            from sofa import VecoTokenizer
            return VecoTokenizer.from_pretrained(model_dir, use_fast=False)
        else:
            # TODO Only support veco & sbert
            raise RuntimeError(f'Unsupported model type: {model_type}')


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.token_cls_tokenizer)
class TokenClassificationPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)

    @type_assert(object, str)
    def __call__(self, data: Union[str, Dict]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        # preprocess the data for the model input
        if isinstance(data, dict):
            data = data[self.first_sequence]
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


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.zero_shot_cls_tokenizer)
class ZeroShotClassificationPreprocessor(NLPPreprocessorBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        self.sequence_length = kwargs.pop('sequence_length', 512)
        super().__init__(model_dir, *args, **kwargs)

    @type_assert(object, str)
    def __call__(self, data, hypothesis_template: str,
                 candidate_labels: list) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if isinstance(data, dict):
            data = data.get(self.first_sequence)

        pairs = [[data, hypothesis_template.format(label)]
                 for label in candidate_labels]

        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.sequence_length,
            return_tensors='pt',
            truncation_strategy='only_first')
        return features
