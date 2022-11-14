# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import logging
import os
import sys
import unicodedata

import json
import regex as re


def clean_string(string):
    replace_mp = {
        ' - ': '-',
        " ' ": "'",
        " n't": "n't",
        " 'm": "'m",
        ' do not': " don't",
        " 's": "'s",
        " 've": "'ve",
        " 're": "'re"
    }
    for k, v in replace_mp.items():
        string = string.replace(k, v)
    return string


class Tokenizer(object):

    def __init__(self, vocab_path, special_tokens=[], tokenizer_type='Bert'):
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'Bert':
            self.spec_convert_dict = {
                '[BOS]': '[unused0]',
                '[EOS]': '[unused1]'
            }
            for token in special_tokens:
                if token not in self.spec_convert_dict and token not in [
                        '[PAD]', '[UNK]'
                ]:
                    self.spec_convert_dict[
                        token] = f'[unused{len(self.spec_convert_dict)}]'
            self.spec_revert_dict = {
                v: k
                for k, v in self.spec_convert_dict.items()
            }
            special_tokens = [
                self.spec_convert_dict.get(tok, tok) for tok in special_tokens
            ]
            self.special_tokens = ('[UNK]', '[SEP]', '[PAD]', '[CLS]',
                                   '[MASK]')
            self.special_tokens += tuple(x for x in special_tokens
                                         if x not in self.special_tokens)

            self._tokenizer = BertTokenizer(
                vocab_path, never_split=self.special_tokens)
            for tok in self.special_tokens:
                assert tok in self._tokenizer.vocab, f"special token '{tok}' is not in the vocabulary"
            self.vocab_size = len(self._tokenizer.vocab)
        elif tokenizer_type == 'GPT2':
            self.spec_convert_dict = {'[UNK]': '<unk>'}
            self.spec_revert_dict = {
                v: k
                for k, v in self.spec_convert_dict.items()
            }
            special_tokens = [
                tok for tok in special_tokens
                if tok not in self.spec_convert_dict
            ]
            vocab_file = os.path.join(vocab_path, 'vocab.json')
            merges_file = os.path.join(vocab_path, 'merges.txt')
            self._tokenizer = GPT2Tokenizer(
                vocab_file, merges_file, special_tokens=special_tokens)
            self.num_specials = len(special_tokens)
            self.vocab_size = len(self._tokenizer)
        else:
            raise ValueError

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        if self.tokenizer_type == 'Bert':
            tokens = [self.spec_convert_dict.get(tok, tok) for tok in tokens]
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            return ids
        else:
            tokens = [self.spec_convert_dict.get(tok, tok) for tok in tokens]
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            ids = [(i + self.num_specials) % self.vocab_size for i in ids]
            return ids

    def convert_ids_to_tokens(self, ids):
        if self.tokenizer_type == 'Bert':
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            tokens = [self.spec_revert_dict.get(tok, tok) for tok in tokens]
            return tokens
        else:
            ids = [(i - self.num_specials) % self.vocab_size for i in ids]
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            tokens = [self.spec_revert_dict.get(tok, tok) for tok in tokens]
            return tokens

    def decode(self, ids, ignore_tokens=[]):
        tokens = self.convert_ids_to_tokens(ids)
        if len(ignore_tokens) > 0:
            ignore_tokens = set(ignore_tokens)
            tokens = [tok for tok in tokens if tok not in ignore_tokens]
        if self.tokenizer_type == 'Bert':
            string = ' '.join(tokens).replace(' ##', '')
        else:
            string = ''.join(tokens)
            string = bytearray([
                self._tokenizer.byte_decoder[c] for c in string
            ]).decode('utf-8')
        string = clean_string(string)
        return string


# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 max_len=None,
                 do_basic_tokenize=True,
                 never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                'model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`'
                .format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                'Token indices sequence length is longer than the specified maximum '
                ' sequence length for this BERT model ({} > {}). Running this'
                ' sequence through BERT will result in indexing errors'.format(
                    len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        tmp = (cp >= 0x4E00 and cp <= 0x9FFF)
        tmp = tmp or (cp >= 0x3400 and cp <= 0x4DBF)
        tmp = tmp or (cp >= 0x20000 and cp <= 0x2A6DF)
        tmp = tmp or (cp >= 0x2A700 and cp <= 0x2B73F)
        tmp = tmp or (cp >= 0x2B740 and cp <= 0x2B81F)
        tmp = tmp or (cp >= 0x2B820 and cp <= 0x2CEAF)
        tmp = tmp or (cp >= 0xF900 and cp <= 0xFAFF)
        tmp = tmp or (cp >= 0x2F800 and cp <= 0x2FA1F)
        if tmp:
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    tmp = (cp >= 33 and cp <= 47)
    tmp = tmp or (cp >= 58 and cp <= 64)
    tmp = tmp or (cp >= 91 and cp <= 96)
    tmp = tmp or (cp >= 123 and cp <= 126)
    if tmp:
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


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
    _chr = unichr if sys.version_info[0] == 2 else chr
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
    cs = [_chr(n) for n in cs]
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


class GPT2Tokenizer(object):
    """
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level BPE
    """

    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 special_tokens=None,
                 max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file, encoding='utf-8'))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {
            v: k
            for k, v in self.special_tokens.items()
        }
        logger.info('Special tokens {}'.format(self.special_tokens))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

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

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[ord(b)] for b in token
                            if ord(b) in self.byte_encoder)
            if token == '':
                continue
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        python_version_3 = isinstance(tokens, str)
        python_version_2 = (
            sys.version_info[0] == 2 and isinstance(tokens, unicode))
        if python_version_3 or python_version_2:
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                'Token indices sequence length is longer than the specified maximum '
                ' sequence length for this OpenAI GPT model ({} > {}). Running this'
                ' sequence through the model will result in indexing errors'.
                format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text
