# Copyright (c) Alibaba, Inc. and its affiliates.

import xml.etree.ElementTree as ET

from .xml_obj import XmlObj


class Syllable(XmlObj):

    def __init__(self):
        self.m_phone_list = []
        self.m_tone = None
        self.m_language = None
        self.m_breaklevel = None

    def pronunciation_text(self):
        return ' '.join([str(phone) for phone in self.m_phone_list])

    def phone_count(self):
        return len(self.m_phone_list)

    def tone_text(self):
        return str(self.m_tone.value)

    def save(self):
        pass

    def load(self):
        pass

    def get_phone_meta(self,
                       phone_name,
                       word_pos,
                       syll_pos,
                       tone_text,
                       single_syllable_word=False):
        #  Special case: word with single syllable, the last phone's word_pos should be "word_end"
        if word_pos == 'word_begin' and syll_pos == 's_end' and single_syllable_word:
            word_pos = 'word_end'
        elif word_pos == 'word_begin' and syll_pos not in [
                's_begin',
                's_both',
        ]:  # FIXME: keep accord with Engine logic
            word_pos = 'word_middle'
        elif word_pos == 'word_end' and syll_pos not in ['s_end', 's_both']:
            word_pos = 'word_middle'
        else:
            pass

        return '{{{}$tone{}${}${}}}'.format(phone_name, tone_text, syll_pos,
                                            word_pos)

    def save_metafile(self, word_pos, single_syllable_word=False):
        syllable_phone_cnt = len(self.m_phone_list)

        meta_line_list = []

        for idx, phone in enumerate(self.m_phone_list):
            if syllable_phone_cnt == 1:
                syll_pos = 's_both'
            elif idx == 0:
                syll_pos = 's_begin'
            elif idx == len(self.m_phone_list) - 1:
                syll_pos = 's_end'
            else:
                syll_pos = 's_middle'
            meta_line_list.append(
                self.get_phone_meta(
                    phone,
                    word_pos,
                    syll_pos,
                    self.tone_text(),
                    single_syllable_word=single_syllable_word,
                ))

        return ' '.join(meta_line_list)


class SyllableList(XmlObj):

    def __init__(self, syllables):
        self.m_syllable_list = syllables

    def __len__(self):
        return len(self.m_syllable_list)

    def __index__(self, index):
        return self.m_syllable_list[index]

    def pronunciation_text(self):
        return ' - '.join([
            syllable.pronunciation_text() for syllable in self.m_syllable_list
        ])

    def tone_text(self):
        return ''.join(
            [syllable.tone_text() for syllable in self.m_syllable_list])

    def save(self, parent_node):
        syllable_node = ET.SubElement(parent_node, 'syllable')
        syllable_node.set('syllcount', str(len(self.m_syllable_list)))

        phone_node = ET.SubElement(syllable_node, 'phone')
        phone_node.text = self.pronunciation_text()

        tone_node = ET.SubElement(syllable_node, 'tone')
        tone_node.text = self.tone_text()

        return

    def load(self):
        pass
