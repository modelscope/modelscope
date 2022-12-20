# Copyright (c) Alibaba, Inc. and its affiliates.

import xml.etree.ElementTree as ET

from .xml_obj import XmlObj


class ScriptItem(XmlObj):

    def __init__(self, phoneset, posset):
        if phoneset is None or posset is None:
            raise Exception('ScriptItem.__init__: phoneset or posset is None')
        self.m_phoneset = phoneset
        self.m_posset = posset

        self.m_id = None
        self.m_text = ''
        self.m_scriptSentence_list = []
        self.m_status = None

    def load(self):
        pass

    def save(self, parent_node):
        utterance_node = ET.SubElement(parent_node, 'utterance')
        utterance_node.set('id', self.m_id)

        text_node = ET.SubElement(utterance_node, 'text')
        text_node.text = self.m_text

        for sentence in self.m_scriptSentence_list:
            sentence.save(utterance_node)

    def save_metafile(self):
        meta_line = self.m_id + '\t'

        for sentence in self.m_scriptSentence_list:
            meta_line += sentence.save_metafile()

        return meta_line
