# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import Any, Dict, List, Optional, Union

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

logger = get_logger()


class TextGenerationPreprocessorBase(Preprocessor):

    def __init__(self,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt='tgt_txt',
                 keep_original_columns=None):
        """The base class for all the text generation task's preprocessors.

        Args:
            mode: The preprocessor mode.
            src_txt: The key for the src text.
            tgt_txt: The key for the tgt text.
            keep_original_columns: Keep original columns and change them to attributes,
                only available when the input is a `dict`, default True
        """
        super().__init__(mode)
        self.src_txt = src_txt
        self.tgt_txt = tgt_txt
        self.keep_original_columns = keep_original_columns

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
        if self.keep_original_columns and isinstance(data, dict):
            for column in self.keep_original_columns:
                output[column] = data[column]
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
                 max_length: int = None,
                 use_fast: bool = None,
                 keep_original_columns=None,
                 **kwargs):
        """The tokenizer preprocessor used in text generation.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            mode: The mode for the preprocessor.
            src_txt: The key of the source sentence.
            tgt_txt: The key of the generated sentence.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            use_fast: Whether to use the fast tokenizer or not.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        if 'first_sequence' in kwargs:
            src_txt = kwargs.pop('first_sequence')
        super().__init__(mode, src_txt, tgt_txt, keep_original_columns)
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     False)
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
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
                labels = self._get_labels_from_tgt(sequence2)
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

    def _get_labels_from_tgt(self, sequence: str) -> torch.Tensor:
        self.nlp_tokenizer.tokenize_kwargs['max_length'] = self.tgt_length
        labels = self.nlp_tokenizer(sequence)['input_ids']
        self.nlp_tokenizer.tokenize_kwargs['max_length'] = self.src_length
        return labels


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_gen_jieba_tokenizer)
class TextGenerationJiebaPreprocessor(TextGenerationPreprocessorBase):
    """The jieba tokenizer preprocessor used in text generation.
    """

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt='tgt_txt',
                 sequence_length: int = 128,
                 use_fast=None,
                 **kwargs):
        from modelscope.models.nlp.gpt3 import JiebaBPETokenizer
        super().__init__(mode, src_txt, tgt_txt, **kwargs)
        self.tokenizer = JiebaBPETokenizer(
            osp.join(model_dir, 'tokenizer.json'))
        self.max_length = sequence_length

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
        if self.mode == ModeKeys.INFERENCE:
            return {
                'input_ids':
                torch.tensor(self.tokenizer.tokenize(sequence1)).unsqueeze_(0)
            }
        # continue write train: | inputs | <sep> |
        # input & output train: | inputs | outputs | <sep> |
        else:
            input_tokens = self.tokenizer.tokenize(sequence1)
            if sequence2 is None:
                return self._only_input(input_tokens)
            else:
                return self._input_and_output(
                    input_tokens, self.tokenizer.tokenize(sequence2))

    def _only_input(self, input_tokens: List[int]) -> Dict[str, Any]:
        prompts_len = len(input_tokens)
        input_tokens.append(self.tokenizer.sep_token)
        tokens = self._truncate(np.asarray(input_tokens))
        return {
            'tokens': tokens[:-1],
            'labels': tokens[1:],
            'prompts_len': min(prompts_len, self.max_length),
        }

    def _input_and_output(self, input_tokens: List[int],
                          output_tokens: List[int]) -> Dict[str, Any]:
        tokens = input_tokens[:]
        tokens.extend(output_tokens)
        tokens.append(self.tokenizer.sep_token)
        inputs_len = len(tokens)
        tokens = self._truncate(np.asarray(tokens))
        return {
            'tokens': tokens[:-1],
            'labels': tokens[1:],
            'prompts_len': min(len(input_tokens), self.max_length),
            'inputs_len': min(inputs_len, self.max_length),
        }

    def _truncate(self, array: np.ndarray) -> np.ndarray:
        if len(array) < self.max_length:
            return np.pad(
                array, (0, self.max_length - len(array)), constant_values=0)
        else:
            return array[:self.max_length]


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_piece)
class TextGenerationSentencePiecePreprocessor(TextGenerationPreprocessorBase):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt=None,
                 **kwargs):
        """

        Args:
            model_dir: The model dir of the sentence piece model.
            mode: The preprocessor mode, currently either mode will have the same behaviour.
            src_txt: The key of input text, if input format is dict.
            tgt_txt: The key of target text, used in training.

        Examples:
            >>> from modelscope.utils.hub import snapshot_download
            >>> from modelscope.preprocessors import TextGenerationSentencePiecePreprocessor
            >>> model_dir = snapshot_download('langboat/mengzi-gpt-neo-base')
            >>> preprocessor = TextGenerationSentencePiecePreprocessor(model_dir)
            >>> print(preprocessor('test word'))
        """
        if 'first_sequence' in kwargs:
            src_txt = kwargs.pop('first_sequence')

        import sentencepiece as spm
        super().__init__(mode, src_txt, tgt_txt, **kwargs)
        self.tokenizer = None
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.model'):
                m_file = osp.join(model_dir, file_name)
                self.tokenizer = spm.SentencePieceProcessor(model_file=m_file)
                break
        assert self.tokenizer is not None, 'Can not find .model file'

    def __call__(self, data: Union[Dict, str], **kwargs):
        text_a, text_b = parse_text_and_label(data, self.mode, self.src_txt,
                                              self.tgt_txt)[0:2]

        return self._tokenize_text(text_a, text_b, **kwargs)

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        return torch.tensor(
            self.tokenizer.encode([sequence1]), dtype=torch.long)

    def decode(self, tokens, **kwargs):
        """Decode the tokens to real text.

        Args:
            tokens: The output tokens from model's `forward` and `generate`

        Returns:
            The actual text.
        """
        return self.tokenizer.decode(tokens)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text2text_gen_preprocessor)
class TextGenerationT5Preprocessor(TextGenerationTransformersPreprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 src_txt='src_txt',
                 tgt_txt='tgt_txt',
                 use_fast: bool = None,
                 **kwargs):
        """The preprocessor for text to text generation task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            src_txt: The key of the first sequence.
            use_fast: Use the fast tokenizer or not.
            mode: The mode for the preprocessor.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        super().__init__(
            model_dir,
            mode=mode,
            src_txt=src_txt,
            tgt_txt=tgt_txt,
            use_fast=use_fast,
            truncation=kwargs.pop('truncation', True),
            padding=kwargs.pop('padding', 'max_length'),
            return_token_type_ids=kwargs.pop('return_token_type_ids', False),
            **kwargs)


SentencePiecePreprocessor = TextGenerationSentencePiecePreprocessor
