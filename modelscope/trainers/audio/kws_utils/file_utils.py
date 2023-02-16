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

remove_str = ['!sil', '(noise)', '(noise', 'noise)', '·', '’']


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def make_pair(wav_lists, trans_lists):
    trans_table = {}
    for line in trans_lists:
        arr = line.strip().replace('\t', ' ').split()
        if len(arr) < 2:
            logger.debug('invalid line in trans file: {}'.format(line.strip()))
            continue

        trans_table[arr[0]] = line.replace(arr[0], '')\
                                  .replace(' ', '')\
                                  .replace('(noise)', '')\
                                  .replace('noise)', '')\
                                  .replace('(noise', '')\
                                  .replace('!sil', '')\
                                  .replace('·', '')\
                                  .replace('’', '').strip()

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


def query_tokens_id(txt, symbol_table, lexicon_table):
    label = tuple()
    tokens = []

    parts = [txt.replace(' ', '').strip()]
    for part in parts:
        for ch in part:
            if ch == ' ':
                ch = '▁'
            tokens.append(ch)

    for ch in tokens:
        if ch in symbol_table:
            label = label + (symbol_table[ch], )
        elif ch in lexicon_table:
            for sub_ch in lexicon_table[ch]:
                if sub_ch in symbol_table:
                    label = label + (symbol_table[sub_ch], )
                else:
                    label = label + (symbol_table['<blk>'], )
        else:
            label = label + (symbol_table['<blk>'], )

    return label
