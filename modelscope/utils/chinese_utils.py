# Copyright (c) Alibaba, Inc. and its affiliates.

import re
import string

from zhconv import convert

CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
ENGLISH_PUNCTUATION = string.punctuation


def is_chinese_char(word: str):
    chinese_punctuations = {
        '，', '。', '；', '：'
        '！', '？', '《', '》', '‘', '’', '“', '”', '（', '）', '【', '】'
    }
    return len(word) == 1 \
        and ('\u4e00' <= word <= '\u9fa5' or word in chinese_punctuations)


def remove_space_between_chinese_chars(decoded_str: str):
    old_word_list = decoded_str.split(' ')
    new_word_list = []
    start = -1
    for i, word in enumerate(old_word_list):
        if is_chinese_char(word):
            if start == -1:
                start = i
        else:
            if start != -1:
                new_word_list.append(''.join(old_word_list[start:i]))
                start = -1
            new_word_list.append(word)
    if start != -1:
        new_word_list.append(''.join(old_word_list[start:]))
    return ' '.join(new_word_list).strip()


# add space for each chinese char
def rebuild_chinese_str(string: str):
    return ' '.join(''.join([
        f' {char} ' if is_chinese_char(char) else char for char in string
    ]).split())


def normalize_chinese_number(text):
    chinese_number = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    new_text = ''
    for x in text:
        if x in '0123456789':
            x = chinese_number[0]
        new_text += x
    new_text = convert(new_text, 'zh-hans')
    return new_text


def pre_chinese(text, max_words):

    text = text.lower().replace(CHINESE_PUNCTUATION,
                                ' ').replace(ENGLISH_PUNCTUATION, ' ')
    text = re.sub(
        r'\s{2,}',
        ' ',
        text,
    )
    text = text.rstrip('\n')
    text = text.strip(' ')[:max_words]
    return text
