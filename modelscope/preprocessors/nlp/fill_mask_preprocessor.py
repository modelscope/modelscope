# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import re
from abc import abstractmethod
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile
from modelscope.utils.hub import get_model_type
from modelscope.utils.nlp import import_external_nltk_data
from .transformers_tokenizer import NLPTokenizer
from .utils import parse_text_and_label


class FillMaskPreprocessorBase(Preprocessor):

    def __init__(self,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 mode: str = ModeKeys.INFERENCE):
        """The base constructor for all the fill-mask preprocessors.

        Args:
            first_sequence: The key of the first sequence.
            second_sequence: The key of the second sequence.
            mode: The mode for the preprocessor.
        """
        super().__init__(mode)
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence

    def __call__(self, data: Union[str, Tuple, Dict],
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (tuple): [sentence1, sentence2]
                sentence1 (str): a sentence

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        text_a, text_b, _ = parse_text_and_label(data, self.mode,
                                                 self.first_sequence,
                                                 self.second_sequence)
        output = self._tokenize_text(text_a, text_b, **kwargs)
        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        return output

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        """Tokenize the text.

        Args:
            sequence1: The first sequence.
            sequence2: The second sequence which may be None.

        Returns:
            The encoded sequence.
        """
        raise NotImplementedError()

    @property
    def mask_id(self):
        """Return the id of the mask token.

        Returns:
            The id of mask token.
        """
        return None

    @abstractmethod
    def decode(self,
               token_ids,
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
        pass


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.fill_mask)
class FillMaskTransformersPreprocessor(FillMaskPreprocessorBase):

    def __init__(self,
                 model_dir: str = None,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 mode: str = ModeKeys.INFERENCE,
                 max_length: int = None,
                 use_fast: bool = None,
                 **kwargs):
        """The preprocessor for fill mask task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            use_fast: Use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(first_sequence, second_sequence, mode)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        return self.nlp_tokenizer(sequence1, sequence2, **kwargs)

    @property
    def mask_id(self):
        """Return the id of the mask token.

        Returns:
            The id of mask token.
        """
        return self.nlp_tokenizer.tokenizer.mask_token_id

    def decode(self,
               token_ids,
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
        return self.nlp_tokenizer.tokenizer.decode(
            token_ids, skip_special_tokens, clean_up_tokenization_spaces,
            **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.fill_mask_ponet)
class FillMaskPoNetPreprocessor(FillMaskPreprocessorBase):

    def __init__(self,
                 model_dir,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 mode: str = ModeKeys.INFERENCE,
                 max_length: int = None,
                 use_fast: bool = None,
                 **kwargs):
        """The tokenizer preprocessor used in PoNet model's MLM task.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            use_fast: Use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 512)
        kwargs.pop('sequence_length', None)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(first_sequence, second_sequence, mode)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, use_fast=use_fast, tokenize_kwargs=kwargs)

        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        self.language = self.cfg.model.get('language', 'en')
        if self.language == 'en':
            from nltk.tokenize import sent_tokenize
            import_external_nltk_data(
                osp.join(model_dir, 'nltk_data'), 'tokenizers/punkt')
        elif self.language in ['zh', 'cn']:

            def sent_tokenize(para):
                para = re.sub(r'([。！!？\?])([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2',
                              para)  # noqa *
                para = para.rstrip()
                return [_ for _ in para.split('\n') if _]
        else:
            raise NotImplementedError

        self.sent_tokenize = sent_tokenize
        self.max_length = kwargs['max_length']

    def __call__(self, data: Union[str, Tuple, Dict],
                 **kwargs) -> Dict[str, Any]:
        text_a, text_b, _ = parse_text_and_label(data, self.mode,
                                                 self.first_sequence,
                                                 self.second_sequence)
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        output = self.nlp_tokenizer(text_a, text_b, **kwargs)
        max_seq_length = self.max_length

        if text_b is None:
            segment_ids = []
            seg_lens = list(
                map(
                    len,
                    self.nlp_tokenizer.tokenizer(
                        self.sent_tokenize(text_a),
                        add_special_tokens=False,
                        truncation=True)['input_ids']))
            segment_id = [0] + sum(
                [[i] * sl for i, sl in enumerate(seg_lens, start=1)], [])
            segment_id = segment_id[:max_seq_length - 1]
            segment_ids.append(segment_id + [segment_id[-1] + 1]
                               * (max_seq_length - len(segment_id)))
            if self.mode == ModeKeys.INFERENCE:
                segment_ids = torch.tensor(segment_ids)
            output['segment_ids'] = segment_ids

        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        return output

    @property
    def mask_id(self):
        """Return the id of the mask token.

        Returns:
            The id of mask token.
        """
        return self.nlp_tokenizer.tokenizer.mask_token_id

    def decode(self,
               token_ids,
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
        return self.nlp_tokenizer.tokenizer.decode(
            token_ids, skip_special_tokens, clean_up_tokenization_spaces,
            **kwargs)
