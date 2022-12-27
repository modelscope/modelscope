# Copyright (c) Alibaba, Inc. and its affiliates.

import xml.etree.ElementTree as ET
from xml.dom import minidom

from .xml_obj import XmlObj


class Script(XmlObj):

    def __init__(self, phoneset, posset):
        self.m_phoneset = phoneset
        self.m_posset = posset
        self.m_items = []

    def save(self, outputXMLPath):
        root = ET.Element('script')

        root.set('uttcount', str(len(self.m_items)))
        root.set('xmlns', 'http://schemas.alibaba-inc.com/tts')
        for item in self.m_items:
            item.save(root)

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(
            indent='  ', encoding='utf-8')
        with open(outputXMLPath, 'wb') as f:
            f.write(xmlstr)

    def save_meta_file(self):
        meta_lines = []

        for item in self.m_items:
            meta_lines.append(item.save_metafile())

        return meta_lines
