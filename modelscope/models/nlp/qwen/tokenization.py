# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Tokenization classes for QWen."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import base64
import logging
import os
import unicodedata
from io import open
from typing import List, Optional, Tuple, Union

import json
import tiktoken
from transformers import AddedToken, PreTrainedTokenizer

from modelscope.utils.logger import get_logger

logger = get_logger()

TIKTOKEN_NAME = 'qwen.tiktoken'


class QWenTokenizer(PreTrainedTokenizer):
    """QWen tokenizer."""
    """NOTE: This tokenizer will not handle special tokens to avoid injection attacks"""

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        merges_file = os.path.join(pretrained_model_name_or_path,
                                   TIKTOKEN_NAME)
        tokenizer = cls(merges_file, *inputs, **kwargs)
        return tokenizer

    def __init__(self,
                 merges_file,
                 errors='replace',
                 max_len=None,
                 unk_token='<|endoftext|>',
                 bos_token='<|endoftext|>',
                 eos_token='<|endoftext|>',
                 pad_token=None,
                 add_prefix_space=False,
                 add_bos_token=False,
                 add_more_sp_tokens=True,
                 **kwargs):
        bos_token = AddedToken(
            bos_token, lstrip=False, rstrip=False) if isinstance(
                bos_token, str) else bos_token
        eos_token = AddedToken(
            eos_token, lstrip=False, rstrip=False) if isinstance(
                eos_token, str) else eos_token
        unk_token = AddedToken(
            unk_token, lstrip=False, rstrip=False) if isinstance(
                unk_token, str) else unk_token
        pad_token = AddedToken(
            pad_token, lstrip=False, rstrip=False) if isinstance(
                pad_token, str) else pad_token
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
        )
        self.add_bos_token = add_bos_token
        self.max_len = max_len if max_len is not None else int(1e12)

        self.errors = errors  # how to handle errors in decoding

        name = 'QWen'
        ENDOFTEXT = '<|endoftext|>'
        IMSTART = '<|im_start|>'
        IMEND = '<|im_end|>'
        if add_more_sp_tokens:
            special_tokens = (ENDOFTEXT, IMSTART, IMEND, '<R>', '<S>', '<X>',
                              '<mask>', '<sep>') + tuple(
                                  [f'<extra_{i}>' for i in range(200)])
        else:
            special_tokens = (ENDOFTEXT, IMSTART, IMEND)

        PAT_STR = (
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

        def load_tiktoken_bpe(tiktoken_bpe_file: str) -> 'dict[bytes, int]':
            contents = open(tiktoken_bpe_file, 'rb').read()
            return {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split()
                                    for line in contents.splitlines() if line)
            }

        mergeable_ranks = load_tiktoken_bpe(merges_file)
        special_tokens = {
            token: index
            for index, token in enumerate(
                special_tokens, start=len(mergeable_ranks))
        }
        self.special_tokens = special_tokens
        enc = tiktoken.Encoding(
            name,
            pat_str=PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        assert len(mergeable_ranks) + len(
            special_tokens
        ) == enc.n_vocab, f'{len(mergeable_ranks) + len(special_tokens)} != {enc.n_vocab} in encoding'

        self.mergeable_ranks = mergeable_ranks
        self.encoder = self.mergeable_ranks
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.tokenizer = enc  # type: tiktoken.Encoding
        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = special_tokens[IMSTART]
        self.im_end_id = special_tokens[IMEND]

    def __len__(self):
        return self.tokenizer.n_vocab

    def get_vocab(self):
        return self.mergeable_ranks

    def convert_tokens_to_ids(self, tokens):
        ids = []
        # Remove support for py2
        if isinstance(tokens, str):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token))
        if len(ids) > self.max_len:
            logger.warning(
                'Token indices sequence length is longer than the specified maximum '
                ' sequence length for this OpenAI GPT model ({} > {}). Running this'
                ' sequence through the model will result in indexing errors'.
                format(len(ids), self.max_len))
        return ids

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, 'qwen.tiktoken')
        with open(file_path, 'w', encoding='utf8') as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode('utf8') + ' ' + str(v) + '\n'
                w.write(line)
        return (file_path, )

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            pair (`str`, *optional*):
                A second sequence to be encoded with the first.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method. See details in
                [`~PreTrainedTokenizerBase.__call__`]

        Returns:
            `List[str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize('NFC', text)
        for t in self.tokenizer.encode_ordinary(text):
            tokens.append(self.decoder[t])
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.
        """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids)
