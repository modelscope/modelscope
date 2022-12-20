# Copyright (c) Alibaba, Inc. and its affiliates.

from .core_types import (PhoneAMType, PhoneAPType, PhoneCVType, PhoneIFType,
                         PhoneUVType)
from .xml_obj import XmlObj


class Phone(XmlObj):

    def __init__(self):
        self.m_id = None
        self.m_name = None
        self.m_cv_type = PhoneCVType.NULL
        self.m_if_type = PhoneIFType.NULL
        self.m_uv_type = PhoneUVType.NULL
        self.m_ap_type = PhoneAPType.NULL
        self.m_am_type = PhoneAMType.NULL
        self.m_bnd = False

    def __str__(self):
        return self.m_name

    def save(self):
        pass

    def load(self, phone_node):
        ns = '{http://schemas.alibaba-inc.com/tts}'

        id_node = phone_node.find(ns + 'id')
        self.m_id = int(id_node.text)

        name_node = phone_node.find(ns + 'name')
        self.m_name = name_node.text

        cv_node = phone_node.find(ns + 'cv')
        self.m_cv_type = PhoneCVType.parse(cv_node.text)

        if_node = phone_node.find(ns + 'if')
        self.m_if_type = PhoneIFType.parse(if_node.text)

        uv_node = phone_node.find(ns + 'uv')
        self.m_uv_type = PhoneUVType.parse(uv_node.text)

        ap_node = phone_node.find(ns + 'ap')
        self.m_ap_type = PhoneAPType.parse(ap_node.text)

        am_node = phone_node.find(ns + 'am')
        self.m_am_type = PhoneAMType.parse(am_node.text)
