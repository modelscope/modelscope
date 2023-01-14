# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections.abc import Mapping

import json
from transformers import AutoTokenizer

from modelscope.metainfo import Models
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = [
    'NLPTokenizer',
]


class NLPTokenizer:

    def __init__(self,
                 model_dir: str = None,
                 model_type=None,
                 use_fast: bool = None,
                 tokenize_kwargs=None):
        """The transformers tokenizer preprocessor base class.

        Any nlp preprocessor which uses the huggingface tokenizer can inherit from this class.

        Args:
            model_dir (str, `optional`): The local path containing the files used to create a preprocessor.
            use_fast (str, `optional`): Use the fast version of tokenizer
            tokenize_kwargs (dict, `optional`): These args will be directly fed into the tokenizer.
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.tokenize_kwargs = tokenize_kwargs
        if self.tokenize_kwargs is None:
            self.tokenize_kwargs = {}
        self._use_fast = use_fast
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.build_tokenizer()
        return self._tokenizer

    @property
    def use_fast(self):
        if self._use_fast is None:
            if self._use_fast is None and self.model_dir is None:
                self._use_fast = False
            elif self._use_fast is None and os.path.isfile(
                    os.path.join(self.model_dir, 'tokenizer_config.json')):
                with open(
                        os.path.join(self.model_dir, 'tokenizer_config.json'),
                        'r',
                        encoding='utf-8') as f:
                    json_config = json.load(f)
                    self._use_fast = json_config.get('use_fast')
            self._use_fast = False if self._use_fast is None else self._use_fast
        return self._use_fast

    def build_tokenizer(self):
        """Build a tokenizer by the model type.

        NOTE: The fast tokenizers have a multi-thread problem, use it carefully.

        Returns:
            The initialized tokenizer.
        """
        # fast version lead to parallel inference failed
        model_type = self.model_type
        model_dir = self.model_dir
        if model_type == Models.deberta_v2:
            from modelscope.models.nlp.deberta_v2 import DebertaV2Tokenizer, DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast if self.use_fast else DebertaV2Tokenizer
            return tokenizer.from_pretrained(
                model_dir) if model_dir is not None else tokenizer()

        if model_type in (Models.structbert, Models.gpt3, Models.palm,
                          Models.plug, Models.megatron_bert):
            from transformers import BertTokenizer, BertTokenizerFast
            tokenizer = BertTokenizerFast if self.use_fast else BertTokenizer
            return tokenizer.from_pretrained(
                model_dir) if model_dir is not None else tokenizer()
        elif model_type == Models.veco:
            from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast
            tokenizer = XLMRobertaTokenizerFast if self.use_fast else XLMRobertaTokenizer
            return tokenizer.from_pretrained(
                model_dir) if model_dir is not None else tokenizer()

        assert model_dir is not None
        return AutoTokenizer.from_pretrained(model_dir, use_fast=self.use_fast)

    def __call__(self, text, text_pair=None, **kwargs):
        kwargs['max_length'] = kwargs.get('max_length',
                                          kwargs.pop('sequence_length', None))
        if kwargs['max_length'] is None:
            kwargs.pop('max_length')
        tokenize_kwargs = {k: v for k, v in self.tokenize_kwargs.items()}
        tokenize_kwargs.update(kwargs)
        kwargs.update(self.tokenize_kwargs)
        return self.tokenizer(text, text_pair, **tokenize_kwargs)

    def get_tokenizer_kwarg(self, key, default_value=None):
        if key in self.tokenize_kwargs:
            return self.tokenize_kwargs[key]
        return self.tokenizer.init_kwargs.get(key, default_value)
