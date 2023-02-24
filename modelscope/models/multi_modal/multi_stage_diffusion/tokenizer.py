# Part of the implementation is borrowed and modified from CLIP, publicly available at https://github.com/openai/CLIP.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import gzip
import html
from functools import lru_cache

import ftfy
import regex as re
import torch
from transformers import AutoTokenizer

__all__ = ['CLIPTokenizer', 'XGLMTokenizer']


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord('!'),
                    ord('~') + 1)) + list(range(
                        ord('¡'),
                        ord('¬') + 1)) + list(range(ord('®'),
                                                    ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):

    def __init__(self, bpe_path):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode('utf-8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            '<|startoftext|>': '<|startoftext|>',
            '<|endoftext|>': '<|endoftext|>'
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>', )
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors='replace').replace('</w>', ' ')
        return text


class CLIPTokenizer(object):
    r"""CLIP tokenizer, adapted from https://github.com/openai/CLIP.
    """

    def __init__(self, bpe_path, length=77):
        self.bpe_path = bpe_path
        self.length = length

        # init tokenizer
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        self.sos_token = self.tokenizer.encoder['<|startoftext|>']
        self.eos_token = self.tokenizer.encoder['<|endoftext|>']
        self.vocab_size = len(self.tokenizer.encoder)

    def __call__(self, sequence):
        if isinstance(sequence, str):
            return torch.LongTensor(self._tokenizer(sequence))
        elif isinstance(sequence, list):
            return torch.LongTensor([self._tokenizer(u) for u in sequence])
        else:
            raise TypeError(
                f'Expected the "sequence" to be a string or a list, but got {type(sequence)}'
            )

    def _tokenizer(self, text):
        tokens = self.tokenizer.encode(text)[:self.length - 2]
        tokens = [self.sos_token] + tokens + [self.eos_token]
        tokens = tokens + [0] * (self.length - len(tokens))
        return tokens


class XGLMTokenizer(object):
    r"""A wrapper of HuggingFace's XGLM tokenizer.
    """

    def __init__(self, model_dir, length=77, **kwargs):
        self.length = length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        _kwargs = {
            'return_tensors': 'pt',
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.length
        }
        _kwargs.update(**kwargs)
        tokens = self.tokenizer(sequence, **_kwargs)
        return tokens.input_ids, tokens.attention_mask
