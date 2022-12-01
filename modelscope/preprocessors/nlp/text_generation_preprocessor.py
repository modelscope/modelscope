# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from modelscope.utils.logger import get_logger
from .transformers_tokenizer import NLPTokenizer
from .utils import parse_text_and_label

logger = get_logger(__name__)


class TextGenerationPreprocessorBase(Preprocessor):

    def __init__(self,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt='tgt_txt'):
        """The base class for all the text generation task's preprocessors.

        Args:
            mode: The preprocessor mode.
            src_txt: The key for the src text.
            tgt_txt: The key for the tgt text.
        """
        super().__init__(mode)
        self.src_txt = src_txt
        self.tgt_txt = tgt_txt

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        """Tokenize the text.

        Args:
            sequence1: The first sequence.
            sequence2: The second sequence which may be None.

        Returns:
            The encoded sequence.
        """
        raise NotImplementedError()

    def __call__(self, data: Union[Dict, str], **kwargs) -> Dict[str, Any]:
        text_a, text_b = parse_text_and_label(data, self.mode, self.src_txt,
                                              self.tgt_txt)[0:2]

        output = self._tokenize_text(text_a, text_b, **kwargs)
        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        return output

    def decode(self, tokens, **kwargs):
        """Decode the tokens to real text.

        Args:
            tokens: The output tokens from model's `forward` and `generate`

        Returns:
            The actual text.
        """
        raise NotImplementedError()


class NLPTokenizerForRoberta(NLPTokenizer):

    def build_tokenizer(self):

        def get_roberta_tokenizer_dir(model_dir: str) -> Optional[str]:
            import os
            for name in os.listdir(model_dir):
                full_name = os.path.join(model_dir, name)
                if 'roberta' in name and os.path.isdir(full_name):
                    return full_name

        roberta_tokenizer_dir = get_roberta_tokenizer_dir(self.model_dir)
        if roberta_tokenizer_dir:
            from transformers import RobertaTokenizer
            return RobertaTokenizer.from_pretrained(
                roberta_tokenizer_dir, do_lower_case=False)
        return super().build_tokenizer()


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_gen_tokenizer)
class TextGenerationTransformersPreprocessor(TextGenerationPreprocessorBase):

    def __init__(self,
                 model_dir: str,
                 tokenizer=None,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt='tgt_txt',
                 sequence_length: int = 128,
                 use_fast: bool = None,
                 **kwargs):
        """The tokenizer preprocessor used in text generation.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            mode: The mode for the preprocessor.
            src_txt: The key of the source sentence.
            tgt_txt: The key of the generated sentence.
            sequence_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            use_fast: Whether to use the fast tokenizer or not.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        if 'first_sequence' in kwargs:
            src_txt = kwargs.pop('first_sequence')
        super().__init__(mode, src_txt, tgt_txt)
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     False)
        kwargs['max_length'] = sequence_length
        self.src_length = kwargs['max_length']
        self.tgt_length = kwargs.pop('target_max_length', kwargs['max_length'])
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizerForRoberta(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)

    def decode(self, tokens, **kwargs):
        """Decode the tokens to real text.

        Args:
            tokens: The output tokens from model's `forward` and `generate`

        Returns:
            The actual text.
        """
        return self.nlp_tokenizer.tokenizer.decode(tokens, **kwargs)

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        """Tokenize the text.

        Args:
            sequence1: The first sequence.
            sequence2: The second sequence which may be None.

        Returns:
            The encoded sequence.
        """
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None

        output = self.nlp_tokenizer(sequence1, **kwargs)
        if self.mode != ModeKeys.INFERENCE:
            if sequence2 is not None:
                self.nlp_tokenizer.tokenize_kwargs[
                    'max_length'] = self.tgt_length
                labels = self.nlp_tokenizer(sequence2)['input_ids']
                self.nlp_tokenizer.tokenize_kwargs[
                    'max_length'] = self.src_length

                src_input_ids = output['input_ids']
                src_attention_mask = output['attention_mask']
            else:
                labels = output['input_ids'][1:]
                src_input_ids = output['input_ids'][:-1]
                src_attention_mask = output['attention_mask'][:-1]

            output = {
                'input_ids': src_input_ids,
                'attention_mask': src_attention_mask,
                'labels': labels,
            }
        return output


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_gen_jieba_tokenizer)
class TextGenerationJiebaPreprocessor(TextGenerationPreprocessorBase):
    """The jieba tokenizer preprocessor used in text generation.
    """

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt=None):
        from modelscope.models.nlp.gpt3 import JiebaBPETokenizer
        super().__init__(mode, src_txt, tgt_txt)
        if self.tgt_txt is not None:
            logger.warn(
                f'TextGenerationJiebaPreprocessor currently does not support training, '
                f'the {self.tgt_txt} of the tgt_txt field will be ignored.')
        self.src_txt = src_txt
        self.tokenizer = JiebaBPETokenizer(
            osp.join(model_dir, 'tokenizer.json'))

    def decode(self, tokens, **kwargs):
        """Decode the tokens to real text.

        Args:
            tokens: The output tokens from model's `forward` and `generate`

        Returns:
            The actual text.
        """
        return self.tokenizer.detokenize(tokens)

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        """Tokenize the text.

        Args:
            sequence1: The first sequence.
            sequence2: The second sequence which may be None.

        Returns:
            The encoded sequence.
        """
        return {
            'input_ids':
            torch.tensor(self.tokenizer.tokenize(sequence1)).unsqueeze_(0)
        }


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text2text_gen_preprocessor)
class TextGenerationT5Preprocessor(TextGenerationTransformersPreprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt='tgt_txt',
                 use_fast: bool = None,
                 sequence_length: int = 128,
                 **kwargs):
        """The preprocessor for text to text generation task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            src_txt: The key of the first sequence.
            use_fast: Use the fast tokenizer or not.
            sequence_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            mode: The mode for the preprocessor.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        super().__init__(
            model_dir,
            mode=mode,
            src_txt=src_txt,
            tgt_txt=tgt_txt,
            sequence_length=sequence_length,
            use_fast=use_fast,
            truncation=kwargs.pop('truncation', True),
            padding=kwargs.pop('padding', 'max_length'),
            return_token_type_ids=kwargs.pop('return_token_type_ids', False),
            **kwargs)
