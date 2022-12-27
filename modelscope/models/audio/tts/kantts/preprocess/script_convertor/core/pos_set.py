# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import xml.etree.ElementTree as ET

from .pos import Pos
from .xml_obj import XmlObj


class PosSet(XmlObj):

    def __init__(self, posset_path):
        self.m_pos_list = []
        self.m_id_map = {}
        self.m_name_map = {}
        self.load(posset_path)

    def load(self, file_path):
        #  alibaba tts xml namespace
        ns = '{http://schemas.alibaba-inc.com/tts}'

        posset_root = ET.parse(file_path).getroot()
        for pos_node in posset_root.findall(ns + 'pos'):
            pos = Pos()
            pos.load(pos_node)
            self.m_pos_list.append(pos)
            if pos.m_id in self.m_id_map:
                logging.error('PosSet.Load: duplicate id: %d', pos.m_id)
            self.m_id_map[pos.m_id] = pos

            if pos.m_name in self.m_name_map:
                logging.error('PosSet.Load duplicate name name: %s',
                              pos.m_name)
            self.m_name_map[pos.m_name] = pos

            if len(pos.m_sub_pos_list) > 0:
                for sub_pos in pos.m_sub_pos_list:
                    self.m_pos_list.append(sub_pos)
                    if sub_pos.m_id in self.m_id_map:
                        logging.error('PosSet.Load: duplicate id: %d',
                                      sub_pos.m_id)
                    self.m_id_map[sub_pos.m_id] = sub_pos

                    if sub_pos.m_name in self.m_name_map:
                        logging.error('PosSet.Load duplicate name name: %s',
                                      sub_pos.m_name)
                    self.m_name_map[sub_pos.m_name] = sub_pos

    def save(self):
        pass
