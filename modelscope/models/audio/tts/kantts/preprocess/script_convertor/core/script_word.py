# Copyright (c) Alibaba, Inc. and its affiliates.

import xml.etree.ElementTree as ET

from .core_types import Language
from .syllable import SyllableList
from .xml_obj import XmlObj


class WrittenWord(XmlObj):

    def __init__(self):
        self.m_name = None
        self.m_POS = None

    def __str__(self):
        return self.m_name

    def load(self):
        pass

    def save(self):
        pass


class WrittenMark(XmlObj):

    def __init__(self):
        self.m_punctuation = None

    def __str__(self):
        return self.m_punctuation

    def load(self):
        pass

    def save(self):
        pass


class SpokenWord(XmlObj):

    def __init__(self):
        self.m_name = None
        self.m_language = None
        self.m_syllable_list = []
        self.m_breakText = '1'
        self.m_POS = '0'

    def __str__(self):
        return self.m_name

    def load(self):
        pass

    def save(self, parent_node):

        word_node = ET.SubElement(parent_node, 'word')

        name_node = ET.SubElement(word_node, 'name')
        name_node.text = self.m_name

        if (len(self.m_syllable_list) > 0
                and self.m_syllable_list[0].m_language != Language.Neutral):
            language_node = ET.SubElement(word_node, 'lang')
            language_node.text = self.m_syllable_list[0].m_language.name

        SyllableList(self.m_syllable_list).save(word_node)

        break_node = ET.SubElement(word_node, 'break')
        break_node.text = self.m_breakText

        POS_node = ET.SubElement(word_node, 'POS')
        POS_node.text = self.m_POS

        return

    def save_metafile(self):
        word_phone_cnt = sum(
            [syllable.phone_count() for syllable in self.m_syllable_list])
        word_syllable_cnt = len(self.m_syllable_list)
        single_syllable_word = word_syllable_cnt == 1
        meta_line_list = []

        for idx, syll in enumerate(self.m_syllable_list):
            if word_phone_cnt == 1:
                word_pos = 'word_both'
            elif idx == 0:
                word_pos = 'word_begin'
            elif idx == len(self.m_syllable_list) - 1:
                word_pos = 'word_end'
            else:
                word_pos = 'word_middle'
            meta_line_list.append(
                syll.save_metafile(
                    word_pos, single_syllable_word=single_syllable_word))

        if self.m_breakText != '0' and self.m_breakText is not None:
            meta_line_list.append('{{#{}$tone_none$s_none$word_none}}'.format(
                self.m_breakText))

        return ' '.join(meta_line_list)


class SpokenMark(XmlObj):

    def __init__(self):
        self.m_breakLevel = None

    def break_level2text(self):
        return '#' + str(self.m_breakLevel.value)

    def __str__(self):
        return self.break_level2text()

    def load(self):
        pass

    def save(self):
        pass
