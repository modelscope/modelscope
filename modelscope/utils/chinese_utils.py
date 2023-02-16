# Copyright (c) Alibaba, Inc. and its affiliates.

import re
import string

from zhconv import convert

CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
ENGLISH_PUNCTUATION = string.punctuation


def remove_space_between_chinese_chars(decoded_str: str):
    old_word_list = decoded_str.split(' ')
    new_word_list = []
    start = -1
    for i, word in enumerate(old_word_list):
        if _is_chinese_str(word):
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
        f' {char} '
        if _is_chinese_char(char) or char in CHINESE_PUNCTUATION else char
        for char in string
    ]).split())


def _is_chinese_str(string: str) -> bool:
    return all(
        _is_chinese_char(cp) or cp in CHINESE_PUNCTUATION
        or cp in ENGLISH_PUNCTUATION or cp for cp in string)


def _is_chinese_char(cp: str) -> bool:
    """Checks whether CP is the codepoint of a CJK character."""
    cp = ord(cp)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True

    return False


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
