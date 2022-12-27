# Copyright (c) Alibaba, Inc. and its affiliates.

from .xml_obj import XmlObj


class Pos(XmlObj):

    def __init__(self):
        self.m_id = None
        self.m_name = None
        self.m_desc = None
        self.m_level = 1
        self.m_parent = None
        self.m_sub_pos_list = []

    def __str__(self):
        return self.m_name

    def save(self):
        pass

    def load(self, pos_node):
        ns = '{http://schemas.alibaba-inc.com/tts}'

        id_node = pos_node.find(ns + 'id')
        self.m_id = int(id_node.text)

        name_node = pos_node.find(ns + 'name')
        self.m_name = name_node.text

        desc_node = pos_node.find(ns + 'desc')
        self.m_desc = desc_node.text

        sub_node = pos_node.find(ns + 'sub')
        if sub_node is not None:
            for sub_pos_node in sub_node.findall(ns + 'pos'):
                sub_pos = Pos()
                sub_pos.load(sub_pos_node)
                sub_pos.m_parent = self
                sub_pos.m_level = self.m_level + 1
                self.m_sub_pos_list.append(sub_pos)

        return
