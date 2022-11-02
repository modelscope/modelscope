# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from abc import ABC
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple, Union

import json
import numpy as np
import torch
from transformers import AutoTokenizer

from modelscope.metainfo import Models
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.base import Preprocessor
from modelscope.utils.constant import ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = [
    'NLPBasePreprocessor',
    'NLPTokenizerPreprocessorBase',
]


class NLPBasePreprocessor(Preprocessor, ABC):

    def __init__(self,
                 model_dir: str,
                 first_sequence=None,
                 second_sequence=None,
                 label=None,
                 label2id=None,
                 mode=ModeKeys.INFERENCE,
                 use_fast=None,
                 **kwargs):
        """The NLP preprocessor base class.

        Args:
            model_dir (str): The local model path
            first_sequence: The key for the first sequence
            second_sequence: The key for the second sequence
            label: The label key
            label2id: An optional label2id mapping, the class will try to call utils.parse_label_mapping
                if this mapping is not supplied.
            mode: Run this preprocessor in either 'train'/'eval'/'inference' mode
            use_fast: use the fast version of tokenizer

        """
        self.model_dir = model_dir
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.label = label

        self.use_fast = use_fast
        if self.use_fast is None and model_dir is None:
            self.use_fast = False
        elif self.use_fast is None and os.path.isfile(
                os.path.join(model_dir, 'tokenizer_config.json')):
            with open(os.path.join(model_dir, 'tokenizer_config.json'),
                      'r') as f:
                json_config = json.load(f)
                self.use_fast = json_config.get('use_fast')
        self.use_fast = False if self.use_fast is None else self.use_fast

        self.label2id = label2id
        if self.label2id is None and model_dir is not None:
            self.label2id = parse_label_mapping(model_dir)
        super().__init__(mode, **kwargs)

    @property
    def mask_id(self):
        """Child preprocessor can override this property to return the id of mask token.

        Returns:
            The id of mask token, default None.
        """
        return None

    def decode(self,
               token_ids: Union[int, List[int], 'np.ndarray', 'torch.Tensor',
                                'tf.Tensor'],
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True,
               **kwargs):
        """Turn the token_ids to real sentence.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
        Returns:
            The real sentence decoded by the preprocessor.
        """
        raise NotImplementedError()


class NLPTokenizerPreprocessorBase(NLPBasePreprocessor):

    def __init__(self,
                 model_dir: str,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 label: str = 'label',
                 label2id: dict = None,
                 mode: str = ModeKeys.INFERENCE,
                 use_fast: bool = None,
                 **kwargs):
        """The NLP tokenizer preprocessor base class.

        Any nlp preprocessor which uses the hf tokenizer can inherit from this class.

        Args:
            model_dir (str): The local model path
            first_sequence: The key for the first sequence
            second_sequence: The key for the second sequence
            label: The key for the label
            label2id: An optional label2id dict.
                If label2id is None, the preprocessor will try to parse label-id mapping from:
                - configuration.json model.label2id/model.id2label
                - config.json label2id/id2label
                - label_mapping.json
            mode: Run this preprocessor in either 'train'/'eval'/'inference' mode, the behavior may be different.
            use_fast: use the fast version of tokenizer
            kwargs: These kwargs will be directly fed into the tokenizer.
        """

        super().__init__(model_dir, first_sequence, second_sequence, label,
                         label2id, mode, use_fast, **kwargs)
        self.model_dir = model_dir
        self.tokenize_kwargs = kwargs
        self.tokenizer = self.build_tokenizer(model_dir)
        logger.info(f'The key of sentence1: {self.first_sequence}, '
                    f'The key of sentence2: {self.second_sequence}, '
                    f'The key of label: {self.label}')
        if self.first_sequence is None:
            logger.warning('[Important] first_sequence attribute is not set, '
                           'this will cause an error if your input is a dict.')

    @property
    def id2label(self):
        """Return the id2label mapping according to the label2id mapping.

        @return: The id2label mapping if exists.
        """
        if self.label2id is not None:
            return {id: label for label, id in self.label2id.items()}
        return None

    def build_tokenizer(self, model_dir):
        """Build a tokenizer by the model type.

        NOTE: This default implementation only returns slow tokenizer, because the fast tokenizers have a
        multi-thread problem.

        Args:
            model_dir:  The local model dir.

        Returns:
            The initialized tokenizer.
        """
        self.is_transformer_based_model = 'lstm' not in model_dir
        # fast version lead to parallel inference failed
        model_type = get_model_type(model_dir)
        if model_type in (Models.structbert, Models.gpt3, Models.palm,
                          Models.plug):
            from modelscope.models.nlp.structbert import SbertTokenizer, SbertTokenizerFast
            tokenizer = SbertTokenizerFast if self.use_fast else SbertTokenizer
            return tokenizer.from_pretrained(model_dir)
        elif model_type == Models.veco:
            from modelscope.models.nlp.veco import VecoTokenizer, VecoTokenizerFast
            tokenizer = VecoTokenizerFast if self.use_fast else VecoTokenizer
            return tokenizer.from_pretrained(model_dir)
        elif model_type == Models.deberta_v2:
            from modelscope.models.nlp.deberta_v2 import DebertaV2Tokenizer, DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast if self.use_fast else DebertaV2Tokenizer
            return tokenizer.from_pretrained(model_dir)
        elif not self.is_transformer_based_model:
            from transformers import BertTokenizer, BertTokenizerFast
            tokenizer = BertTokenizerFast if self.use_fast else BertTokenizer
            return tokenizer.from_pretrained(model_dir)
        else:
            return AutoTokenizer.from_pretrained(
                model_dir, use_fast=self.use_fast)

    def __call__(self, data: Union[str, Tuple, Dict]) -> Dict[str, Any]:
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

        text_a, text_b, labels = self.parse_text_and_label(data)
        output = self.tokenizer(
            text_a,
            text_b,
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None,
            **self.tokenize_kwargs)
        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        self.labels_to_id(labels, output)
        return output

    def parse_text_and_label(self, data):
        """Parse the input and return the sentences and labels.

        When input type is tuple or list and its size is 2:
        If the pair param is False, data will be parsed as the first_sentence and the label,
        else it will be parsed as the first_sentence and the second_sentence.

        Args:
            data: The input data.

        Returns:
            The sentences and labels tuple.
        """
        text_a, text_b, labels = None, None, None
        if isinstance(data, str):
            text_a = data
        elif isinstance(data, tuple) or isinstance(data, list):
            if len(data) == 3:
                text_a, text_b, labels = data
            elif len(data) == 2:
                if self._mode == ModeKeys.INFERENCE:
                    text_a, text_b = data
                else:
                    text_a, labels = data
        elif isinstance(data, Mapping):
            text_a = data.get(self.first_sequence)
            text_b = data.get(self.second_sequence)
            labels = data.get(self.label)

        return text_a, text_b, labels

    def labels_to_id(self, labels, output):
        """Turn the labels to id with the type int or float.

        If the original label's type is str or int, the label2id mapping will try to convert it to the final label.
        If the original label's type is float, or the label2id mapping does not exist,
        the original label will be returned.

        Args:
            labels: The input labels.
            output: The label id.

        Returns:
            The final labels.
        """

        def label_can_be_mapped(label):
            return isinstance(label, str) or isinstance(label, int)

        try:
            if isinstance(labels, (tuple, list)) and all([label_can_be_mapped(label) for label in labels]) \
                    and self.label2id is not None:
                output[OutputKeys.LABELS] = [
                    self.label2id[label]
                    if label in self.label2id else self.label2id[str(label)]
                    for label in labels
                ]
            elif label_can_be_mapped(labels) and self.label2id is not None:
                output[OutputKeys.LABELS] = self.label2id[
                    labels] if labels in self.label2id else self.label2id[str(
                        labels)]
            elif labels is not None:
                output[OutputKeys.LABELS] = labels
        except KeyError as e:
            logger.error(
                f'Label {labels} cannot be found in the label mapping {self.label2id},'
                f'which comes from the user input or the configuration files. '
                f'Please consider matching your labels with this mapping.')
            raise e
