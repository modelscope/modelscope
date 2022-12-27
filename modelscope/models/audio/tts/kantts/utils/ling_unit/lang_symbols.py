# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import xml.etree.ElementTree as ET

from modelscope.models.audio.tts.kantts.preprocess.languages import languages
from modelscope.utils.logger import get_logger

logging = get_logger()

syllable_flags = [
    's_begin',
    's_end',
    's_none',
    's_both',
    's_middle',
]

word_segments = [
    'word_begin',
    'word_end',
    'word_middle',
    'word_both',
    'word_none',
]


def parse_phoneset(phoneset_file):
    """Parse a phoneset file and return a list of symbols.
    Args:
        phoneset_file (str): Path to the phoneset file.

    Returns:
        list: A list of phones.
    """
    ns = '{http://schemas.alibaba-inc.com/tts}'

    phone_lst = []
    phoneset_root = ET.parse(phoneset_file).getroot()
    for phone_node in phoneset_root.findall(ns + 'phone'):
        phone_lst.append(phone_node.find(ns + 'name').text)

    for i in range(1, 5):
        phone_lst.append('#{}'.format(i))

    return phone_lst


def parse_tonelist(tonelist_file):
    """Parse a tonelist file and return a list of tones.
    Args:
        tonelist_file (str): Path to the tonelist file.

    Returns:
        dict: A dictionary of tones.
    """
    tone_lst = []
    with open(tonelist_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        tone = line.strip()
        if tone != '':
            tone_lst.append('tone{}'.format(tone))
        else:
            tone_lst.append('tone_none')

    return tone_lst


def get_language_symbols(language, language_dir):
    """Get symbols of a language.
    Args:
        language (str): Language name.
    """
    language_dict = languages.get(language, None)
    if language_dict is None:
        logging.error('Language %s not supported. Using PinYin as default',
                      language)
        language_dict = languages['PinYin']
        language = 'PinYin'

    language_dir = os.path.join(language_dir, language)
    phoneset_file = os.path.join(language_dir, language_dict['phoneset_path'])
    tonelist_file = os.path.join(language_dir, language_dict['tonelist_path'])
    phones = parse_phoneset(phoneset_file)
    tones = parse_tonelist(tonelist_file)

    return phones, tones, syllable_flags, word_segments
