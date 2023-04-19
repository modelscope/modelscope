# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

import re

from modelscope.utils.logger import get_logger

logger = get_logger()

symbol_str = '[’!"#$%&\'()*+,-./:;<>=?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'


def split_mixed_label(input_str):
    tokens = []
    s = input_str.lower()
    while len(s) > 0:
        match = re.match(r'[A-Za-z!?,<>()\']+', s)
        if match is not None:
            word = match.group(0)
        else:
            word = s[0:1]
        tokens.append(word)
        s = s.replace(word, '', 1).strip(' ')
    return tokens


def space_mixed_label(input_str):
    splits = split_mixed_label(input_str)
    space_str = ''.join(f'{sub} ' for sub in splits)
    return space_str.strip()


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            if line.strip() != '':
                lists.append(line.strip())
    return lists


def make_pair(wav_lists, trans_lists):
    trans_table = {}
    for line in trans_lists:
        arr = line.strip().replace('\t', ' ').split()
        if len(arr) < 2:
            logger.debug('invalid line in trans file: {}'.format(line.strip()))
            continue

        trans_table[arr[0]] = line.replace(arr[0], '').strip()

    lists = []
    for line in wav_lists:
        arr = line.strip().replace('\t', ' ').split()
        if len(arr) == 2 and arr[0] in trans_table:
            lists.append(
                dict(
                    key=arr[0],
                    txt=trans_table[arr[0]],
                    wav=arr[1],
                    sample_rate=16000))
        else:
            logger.debug("can't find corresponding trans for key: {}".format(
                arr[0]))
            continue

    return lists


def read_token(token_file):
    tokens_table = {}
    with open(token_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            tokens_table[arr[0]] = int(arr[1]) - 1
    fin.close()
    return tokens_table


def read_lexicon(lexicon_file):
    lexicon_table = {}
    with open(lexicon_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().replace('\t', ' ').split()
            assert len(arr) >= 2
            lexicon_table[arr[0]] = arr[1:]
    fin.close()
    return lexicon_table


def query_token_set(txt, symbol_table, lexicon_table):
    tokens_str = tuple()
    tokens_idx = tuple()

    parts = split_mixed_label(txt)
    for part in parts:
        if part == '!sil' or part == '(sil)' or part == '<sil>':
            tokens_str = tokens_str + ('!sil', )
        elif part == '<blk>' or part == '<blank>':
            tokens_str = tokens_str + ('<blk>', )
        elif part == '(noise)' or part == 'noise)' or part == '(noise' or part == '<noise>':
            tokens_str = tokens_str + ('<GBG>', )
        elif part in symbol_table:
            tokens_str = tokens_str + (part, )
        elif part in lexicon_table:
            for ch in lexicon_table[part]:
                tokens_str = tokens_str + (ch, )
        else:
            # case with symbols or meaningless english letter combination
            part = re.sub(symbol_str, '', part)
            for ch in part:
                tokens_str = tokens_str + (ch, )

    for ch in tokens_str:
        if ch in symbol_table:
            tokens_idx = tokens_idx + (symbol_table[ch], )
        elif ch == '!sil':
            if 'sil' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['sil'], )
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
        elif ch == '<GBG>':
            if '<GBG>' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['<GBG>'], )
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
        else:
            if '<GBG>' in symbol_table:
                tokens_idx = tokens_idx + (symbol_table['<GBG>'], )
                logger.info(
                    f'\'{ch}\' is not in token set, replace with <GBG>')
            else:
                tokens_idx = tokens_idx + (symbol_table['<blk>'], )
                logger.info(
                    f'\'{ch}\' is not in token set, replace with <blk>')

    return tokens_str, tokens_idx


def query_token_list(txt, symbol_table, lexicon_table):
    tokens_str = []
    tokens_idx = []

    parts = split_mixed_label(txt)
    for part in parts:
        if part == '!sil' or part == '(sil)' or part == '<sil>':
            tokens_str.append('!sil')
        elif part == '<blk>' or part == '<blank>':
            tokens_str.append('<blk>')
        elif part == '(noise)' or part == 'noise)' or part == '(noise' or part == '<noise>':
            tokens_str.append('<GBG>')
        elif part in symbol_table:
            tokens_str.append(part)
        elif part in lexicon_table:
            for ch in lexicon_table[part]:
                tokens_str.append(ch)
        else:
            # case with symbols or meaningless english letter combination
            part = re.sub(symbol_str, '', part)
            for ch in part:
                tokens_str.append(ch)

    for ch in tokens_str:
        if ch in symbol_table:
            tokens_idx.append(symbol_table[ch])
        elif ch == '!sil':
            if 'sil' in symbol_table:
                tokens_idx.append(symbol_table['sil'])
            else:
                tokens_idx.append(symbol_table['<blk>'])
        elif ch == '<GBG>':
            if '<GBG>' in symbol_table:
                tokens_idx.append(symbol_table['<GBG>'])
            else:
                tokens_idx.append(symbol_table['<blk>'])
        else:
            if '<GBG>' in symbol_table:
                tokens_idx.append(symbol_table['<GBG>'])
                logger.info(
                    f'\'{ch}\' is not in token set, replace with <GBG>')
            else:
                tokens_idx.append(symbol_table['<blk>'])
                logger.info(
                    f'\'{ch}\' is not in token set, replace with <blk>')

    return tokens_str, tokens_idx


def tokenize(data_list, symbol_table, lexicon_table):
    for sample in data_list:
        assert 'txt' in sample
        txt = sample['txt'].strip()
        strs, indexs = query_token_list(txt, symbol_table, lexicon_table)
        sample['tokens'] = strs
        sample['txt'] = indexs

    return data_list
