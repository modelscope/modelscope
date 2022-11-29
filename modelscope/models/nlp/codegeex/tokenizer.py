# Copyright (c) 2022 Zhipu.AI
from typing import List, Union

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2TokenizerFast


def encode_whitespaces(text, start_extra_id: int, max_len: int):
    """ Encode whitespaces to extra tokens in GPT-J.

    >>> encode_whitespaces('a\\n  b\\n   c', 10, 10)
    'a\\n<|extratoken_10|>b\\n<|extratoken_11|>c'
    """

    def push_acc_space(acc_len: int, text: str):
        if acc_len == 0:
            return text
        if acc_len == 1:
            return text + ' '
        assert acc_len <= max_len, f'Max whitespace run length {max_len}, but found {acc_len}'
        extra_id = start_extra_id - 2 + acc_len
        extra_token = f'<|extratoken_{extra_id}|>'
        return text + extra_token

    acc_len = 0
    res = ''
    for ch in text:
        if ch == ' ':
            acc_len += 1
            if acc_len == max_len:
                res = push_acc_space(acc_len, res)
                acc_len = 0
        else:
            res = push_acc_space(acc_len, res)
            acc_len = 0
            res = res + ch

    res = push_acc_space(acc_len, res)

    return res


def decode_whitespaces(text: str, start_extra_id: int, max_len: int):
    """ Decode the whitespace-encoded strings produced by encode_whitespace.

    >>> text = 'a\\n  b\\n   c'
    >>> s, l = 10, 10
    >>> text == decode_whitespaces(encode_whitespaces(text, s, l), s, l)
    True
    """
    for l in range(2, max_len + 1):  # noqa
        token_id = start_extra_id - 2 + l
        token = f'<|extratoken_{token_id}|>'
        text = text.replace(token, ' ' * l)
    return text


class Code13BDictionary(object):

    def __init__(
        self,
        dict_file: str,
        extra_token_ids: List[str] = None,
        pad_to_vocab_size: int = -1,
    ):
        self._idx = dict()
        self._count = dict()
        self._num_symbols = 0
        self._symbols = []

        self._add_symbol('<s>', 0)
        self._add_symbol('<pad>', 0)
        self._add_symbol('</s>', 0)
        self._add_symbol('<unk>', 0)
        self._load_dict(dict_file)

        if extra_token_ids is None:
            extra_token_ids = [str(x) for x in range(50257, 50400)
                               ]  # follows GPT-J settings

        for token_id in extra_token_ids:
            self._add_symbol(token_id, 0)

        if pad_to_vocab_size > 0:
            self._pad_to_vocab_size(pad_to_vocab_size)

    def _pad_to_vocab_size(self, vocab_size: int):
        num_pad = vocab_size - len(self)
        if num_pad <= 0:
            return
        for i in range(1, num_pad + 1):
            self._add_symbol('vocab_pad_token{}'.format(i), 0)

    def _load_dict(self, dict_file: str):
        with open(dict_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '' or line.startswith('#'):
                    continue
                sym, count = line.split()
                self._add_symbol(sym, int(count))

    def _add_symbol(self, sym: str, count: int):
        self._idx[sym] = self._num_symbols
        self._count[sym] = count
        self._symbols.append(sym)
        self._num_symbols += 1

    def __len__(self):
        return self._num_symbols

    def index(self, sym: str):
        return self._idx[sym]

    def string(self, idx: int):
        return self._symbols[idx]

    def map_token(self, token: Union[int, str]):
        if isinstance(token, int):
            token = str(token)
        return self.index(token)

    def map_tokens(self, tokens):
        return [self.map_token(token) for token in tokens]

    def decode_tokens(self, tokens):
        decoded = [
            '50256' if token == 50256 else self.string(token)
            for token in tokens
        ]
        return [int(x) for x in decoded if not x.startswith('vocab_pad_token')]


class CodeGeeXTokenizer(object):

    def __init__(
        self,
        tokenizer: GPT2TokenizerFast = None,
        tokenizer_path: str = 'EleutherAI/gpt-j-6B',
        start_extra_id: int = 10,
        max_len: int = 10,
        mode='codegeex-13b',
        dict_file: str = None,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            tokenizer_path)
        if mode not in ['codegeex-13b', 'codegeex-python-13b']:
            raise ValueError(
                f"Invalid mode {mode}, choose from ['codegeex-13b', 'codegeex-python-13b']"
            )
        self.start_extra_id = start_extra_id
        self.max_len = max_len
        self.mode = mode
        if dict_file is not None:
            self.code_dict = Code13BDictionary(
                dict_file, pad_to_vocab_size=51200
            ) if self.mode == 'codegeex-python-13b' else None
        else:
            self.code_dict = None
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode_code(self, code: str):
        if self.mode == 'codegeex-13b':
            code = encode_whitespaces(code, self.start_extra_id, self.max_len)
            input_ids = self.tokenizer(
                code, is_split_into_words=False).input_ids

        elif self.mode == 'codegeex-python-13b':
            code = encode_whitespaces(code, self.start_extra_id, self.max_len)
            input_ids = self.code_dict.map_tokens(self.tokenizer.encode(code))
            input_ids = torch.LongTensor(input_ids).reshape(1, -1)

        return input_ids

    def decode_code(self, input_ids):
        if self.mode == 'codegeex-13b':
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            output_code = decode_whitespaces(text, self.start_extra_id,
                                             self.max_len)
        elif self.mode == 'codegeex-python-13b':
            input_ids = [self.code_dict.decode_tokens(input_ids.tolist()[0])]
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            output_code = decode_whitespaces(text, self.start_extra_id,
                                             self.max_len)

        return output_code
