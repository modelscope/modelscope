# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional

import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from modelscope.utils.logger import get_logger
from .transformers_tokenizer import NLPTokenizer

logger = get_logger()


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_embedding)
class SentenceEmbeddingTransformersPreprocessor(Preprocessor):
    """The tokenizer preprocessor used in sentence embedding.
    """

    def __init__(self,
                 model_dir: str,
                 first_sequence='source_sentence',
                 second_sequence='sentences_to_compare',
                 mode=ModeKeys.INFERENCE,
                 use_fast: bool = True,
                 max_length: int = None,
                 **kwargs):
        """The preprocessor for sentence embedding task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            first_sequence: The key of the first sequence.
            second_sequence: The key of the second sequence.
            mode: The mode for the preprocessor.
            use_fast: Use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        model_type = None
        self.max_length = max_length
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        # we could add `boq/bod` token/prompt and `eoq/eod` token if they exist when tokenizing.
        for k in ('boq', 'eoq', 'bod', 'eod'):
            setattr(self, k, kwargs.pop(k, None))
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        super().__init__(mode=mode)
        tokenizer = self.nlp_tokenizer.tokenizer
        # For tokenizers like bloom
        if tokenizer.padding_side != 'right':
            # weighted mean pooling need pad right
            logger.warning(
                f'Change tokenizer.padding_side from {tokenizer.padding_side} to right'
            )
            tokenizer.padding_side = 'right'
        # For decoder-only tokenizers
        if tokenizer.pad_token is None:
            logger.warning(
                f'Set tokenizer.pad_token as eos_token {tokenizer.eos_token}')
            tokenizer.pad_token = tokenizer.eos_token
        # Currently eos is single token, we can extend to prompt later.
        for k in ('eoq', 'eod'):
            v = getattr(self, k, None)
            if v is not None:
                v = tokenizer.convert_tokens_to_ids(v)
            setattr(self, k + '_id', v)
        self.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def __call__(self,
                 data: Dict,
                 padding=True,
                 truncation=True,
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data Dict:
                keys: the source sentence and the sentences to compare
                values: list of sentences
                Example:
                    {"source_sentence": ["how long it take to get a master's degree"],
                     "sentences_to_compare": ["On average, students take about 18 to 24 months
                     to complete a master's degree.",
                     "On the other hand, some students prefer to go at a slower pace
                     and choose to take several years to complete their studies.",
                     "It can take anywhere from two semesters"]}
        Returns:
            Dict[str, Any]: the preprocessed data
        """
        source_sentences = data[self.first_sequence]
        if self.second_sequence in data:
            if isinstance(source_sentences[0], list):
                source_sentences = [source_sentences[0]]
            compare_sentences = data[self.second_sequence]
        else:
            compare_sentences = None
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        query_inputs = self.tokenize(
            source_sentences,
            is_query=True,
            padding=padding,
            truncation=truncation,
            **kwargs)
        tokenized_inputs = {'query': query_inputs, 'docs': None}
        if compare_sentences is not None and len(compare_sentences) > 0:
            tokenized_inputs['docs'] = self.tokenize(
                compare_sentences,
                is_query=kwargs.get('symmetric', False),
                padding=padding,
                truncation=truncation,
                **kwargs)
        return tokenized_inputs

    def tokenize(self, texts, is_query=True, return_tensors=None, **kwargs):
        """Tokenize raw texts, add `boq/bod` token/prompt and `eoq/eod` token if they exist.

        Args:
            `texts` List[str]: texts to tokenize,
                Example:
                    ["how long it take to get a master's degree"]
            `is_query` bool: whether the input text(s) is query.
            `return_tensors` str: the `return_tensors` argument to tokenizer.
        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if is_query:
            bos, eos_id = self.boq, self.eoq_id
        else:
            bos, eos_id = self.bod, self.eod_id
        if bos is not None:
            # bos can be prompt
            texts = [bos + t for t in texts]
        encoding = self.nlp_tokenizer(
            texts, return_tensors=return_tensors, **kwargs)
        if eos_id is not None:
            if return_tensors == 'pt':
                self.add_eos_pt(encoding, eos_id)
            else:
                self.add_eos(encoding, eos_id)
        return encoding

    def add_eos_pt(self, encoding: Dict[str, torch.Tensor], eos: int):
        """Add `eos` token id to the end of each sequence."""
        input_ids, attn_mask = encoding['input_ids'], encoding[
            'attention_mask']
        batch = torch.arange(input_ids.size(0))
        length = attn_mask.sum(-1)

        if input_ids.size(1) < self.max_length:
            ones = input_ids.new_ones(input_ids.size(0), 1)
            attn_mask = torch.cat((ones, attn_mask), dim=1)
            padding = ones * self.pad_id
            input_ids = torch.cat((input_ids, padding), dim=1)
            eos_index = length
        else:
            eos_index = torch.clamp(length, max=self.max_length - 1)
            attn_mask[batch, eos_index] = 1
        input_ids[batch, eos_index] = eos
        encoding['input_ids'], encoding[
            'attention_mask'] = input_ids, attn_mask
        return

    def add_eos(self, encoding: Dict[str, list], eos: int):
        """Add `eos` token id to the end of each sequence."""
        for ids, mask in zip(encoding['input_ids'],
                             encoding['attention_mask']):
            if len(mask) < self.max_length:
                ids.append(eos)
                mask.append(1)
            else:
                last = min(sum(mask), self.max_length - 1)
                ids[last] = eos
                mask[last] = 1
        return
