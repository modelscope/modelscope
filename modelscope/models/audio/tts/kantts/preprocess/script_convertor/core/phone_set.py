# Copyright (c) Alibaba, Inc. and its affiliates.

import xml.etree.ElementTree as ET

from modelscope.utils.logger import get_logger
from .phone import Phone
from .xml_obj import XmlObj

logging = get_logger()


class PhoneSet(XmlObj):

    def __init__(self, phoneset_path):
        self.m_phone_list = []
        self.m_id_map = {}
        self.m_name_map = {}
        self.load(phoneset_path)

    def load(self, file_path):
        #  alibaba tts xml namespace
        ns = '{http://schemas.alibaba-inc.com/tts}'

        phoneset_root = ET.parse(file_path).getroot()
        for phone_node in phoneset_root.findall(ns + 'phone'):
            phone = Phone()
            phone.load(phone_node)
            self.m_phone_list.append(phone)
            if phone.m_id in self.m_id_map:
                logging.error('PhoneSet.Load: duplicate id: %d', phone.m_id)
            self.m_id_map[phone.m_id] = phone

            if phone.m_name in self.m_name_map:
                logging.error('PhoneSet.Load duplicate name name: %s',
                              phone.m_name)
            self.m_name_map[phone.m_name] = phone

    def save(self):
        pass
