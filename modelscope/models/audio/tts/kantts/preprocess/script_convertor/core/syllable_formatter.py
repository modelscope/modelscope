# Copyright (c) Alibaba, Inc. and its affiliates.

import re

from modelscope.utils.logger import get_logger
from .core_types import Language, PhoneCVType, Tone
from .syllable import Syllable
from .utils import NgBreakPattern

logging = get_logger()


class DefaultSyllableFormatter:

    def __init__(self):
        return

    def format(self, phoneset, pronText, syllable_list):
        logging.warning('Using DefaultSyllableFormatter dry run: %s', pronText)
        return True


RegexNg2en = re.compile(NgBreakPattern)
RegexQingSheng = re.compile(r'([1-5]5)')
RegexPron = re.compile(r'(?P<Pron>[a-z]+)(?P<Tone>[1-6])')


class ZhCNSyllableFormatter:

    def __init__(self, sy2ph_map):
        self.m_sy2ph_map = sy2ph_map

    def normalize_pron(self, pronText):
        #  Replace Qing Sheng
        newPron = pronText.replace('6', '2')
        newPron = re.sub(RegexQingSheng, '5', newPron)

        #  FIXME(Jin): ng case overrides newPron
        match = RegexNg2en.search(newPron)
        if match:
            newPron = 'en' + match.group('break')

        return newPron

    def format(self, phoneset, pronText, syllable_list):
        if phoneset is None or syllable_list is None or pronText is None:
            logging.error('ZhCNSyllableFormatter.Format: invalid input')
            return False
        pronText = self.normalize_pron(pronText)

        if pronText in self.m_sy2ph_map:
            phone_list = self.m_sy2ph_map[pronText].split(' ')
            if len(phone_list) == 3:
                syll = Syllable()
                for phone in phone_list:
                    syll.m_phone_list.append(phone)
                syll.m_tone = Tone.parse(
                    pronText[-1])  # FIXME(Jin): assume tone is the last char
                syll.m_language = Language.ZhCN
                syllable_list.append(syll)
                return True
            else:
                logging.error(
                    'ZhCNSyllableFormatter.Format: invalid pronText: %s',
                    pronText)
                return False
        else:
            logging.error(
                'ZhCNSyllableFormatter.Format: syllable to phone map missing key: %s',
                pronText,
            )
            return False


class PinYinSyllableFormatter:

    def __init__(self, sy2ph_map):
        self.m_sy2ph_map = sy2ph_map

    def normalize_pron(self, pronText):
        newPron = pronText.replace('6', '2')
        newPron = re.sub(RegexQingSheng, '5', newPron)

        #  FIXME(Jin): ng case overrides newPron
        match = RegexNg2en.search(newPron)
        if match:
            newPron = 'en' + match.group('break')

        return newPron

    def format(self, phoneset, pronText, syllable_list):
        if phoneset is None or syllable_list is None or pronText is None:
            logging.error('PinYinSyllableFormatter.Format: invalid input')
            return False
        pronText = self.normalize_pron(pronText)

        match = RegexPron.search(pronText)

        if match:
            pron = match.group('Pron')
            tone = match.group('Tone')
        else:
            logging.error(
                'PinYinSyllableFormatter.Format: pronunciation is not valid: %s',
                pronText,
            )
            return False

        if pron in self.m_sy2ph_map:
            phone_list = self.m_sy2ph_map[pron].split(' ')
            if len(phone_list) in [1, 2]:
                syll = Syllable()
                for phone in phone_list:
                    syll.m_phone_list.append(phone)
                syll.m_tone = Tone.parse(tone)
                syll.m_language = Language.PinYin
                syllable_list.append(syll)
                return True
            else:
                logging.error(
                    'PinYinSyllableFormatter.Format: invalid phone: %s', pron)
                return False
        else:
            logging.error(
                'PinYinSyllableFormatter.Format: syllable to phone map missing key: %s',
                pron,
            )
            return False


class ZhHKSyllableFormatter:

    def __init__(self, sy2ph_map):
        self.m_sy2ph_map = sy2ph_map

    def format(self, phoneset, pronText, syllable_list):
        if phoneset is None or syllable_list is None or pronText is None:
            logging.error('ZhHKSyllableFormatter.Format: invalid input')
            return False

        match = RegexPron.search(pronText)
        if match:
            pron = match.group('Pron')
            tone = match.group('Tone')
        else:
            logging.error(
                'ZhHKSyllableFormatter.Format: pronunciation is not valid: %s',
                pronText)
            return False

        if pron in self.m_sy2ph_map:
            phone_list = self.m_sy2ph_map[pron].split(' ')
            if len(phone_list) in [1, 2]:
                syll = Syllable()
                for phone in phone_list:
                    syll.m_phone_list.append(phone)
                syll.m_tone = Tone.parse(tone)
                syll.m_language = Language.ZhHK
                syllable_list.append(syll)
                return True
            else:
                logging.error(
                    'ZhHKSyllableFormatter.Format: invalid phone: %s', pron)
                return False
        else:
            logging.error(
                'ZhHKSyllableFormatter.Format: syllable to phone map missing key: %s',
                pron,
            )
            return False


class WuuShanghaiSyllableFormatter:

    def __init__(self, sy2ph_map):
        self.m_sy2ph_map = sy2ph_map

    def format(self, phoneset, pronText, syllable_list):
        if phoneset is None or syllable_list is None or pronText is None:
            logging.error('WuuShanghaiSyllableFormatter.Format: invalid input')
            return False

        match = RegexPron.search(pronText)
        if match:
            pron = match.group('Pron')
            tone = match.group('Tone')
        else:
            logging.error(
                'WuuShanghaiSyllableFormatter.Format: pronunciation is not valid: %s',
                pronText,
            )
            return False

        if pron in self.m_sy2ph_map:
            phone_list = self.m_sy2ph_map[pron].split(' ')
            if len(phone_list) in [1, 2]:
                syll = Syllable()
                for phone in phone_list:
                    syll.m_phone_list.append(phone)
                syll.m_tone = Tone.parse(tone)
                syll.m_language = Language.WuuShanghai
                syllable_list.append(syll)
                return True
            else:
                logging.error(
                    'WuuShanghaiSyllableFormatter.Format: invalid phone: %s',
                    pron)
                return False
        else:
            logging.error(
                'WuuShanghaiSyllableFormatter.Format: syllable to phone map missing key: %s',
                pron,
            )
            return False


class SichuanSyllableFormatter:

    def __init__(self, sy2ph_map):
        self.m_sy2ph_map = sy2ph_map

    def format(self, phoneset, pronText, syllable_list):
        if phoneset is None or syllable_list is None or pronText is None:
            logging.error('SichuanSyllableFormatter.Format: invalid input')
            return False

        match = RegexPron.search(pronText)
        if match:
            pron = match.group('Pron')
            tone = match.group('Tone')
        else:
            logging.error(
                'SichuanSyllableFormatter.Format: pronunciation is not valid: %s',
                pronText,
            )
            return False

        if pron in self.m_sy2ph_map:
            phone_list = self.m_sy2ph_map[pron].split(' ')
            if len(phone_list) in [1, 2]:
                syll = Syllable()
                for phone in phone_list:
                    syll.m_phone_list.append(phone)
                syll.m_tone = Tone.parse(tone)
                syll.m_language = Language.Sichuan
                syllable_list.append(syll)
                return True
            else:
                logging.error(
                    'SichuanSyllableFormatter.Format: invalid phone: %s', pron)
                return False
        else:
            logging.error(
                'SichuanSyllableFormatter.Format: syllable to phone map missing key: %s',
                pron,
            )
            return False


class EnXXSyllableFormatter:

    def __init__(self, language):
        self.m_f2t_map = None
        self.m_language = language

    def normalize_pron(self, pronText):
        newPron = pronText.replace('#', '.')
        newPron = (
            newPron.replace('03',
                            '0').replace('13',
                                         '1').replace('23',
                                                      '2').replace('3', ''))
        newPron = newPron.replace('2', '0')

        return newPron

    def format(self, phoneset, pronText, syllable_list):
        if phoneset is None or syllable_list is None or pronText is None:
            logging.error('EnXXSyllableFormatter.Format: invalid input')
            return False
        pronText = self.normalize_pron(pronText)

        syllables = [ele.strip() for ele in pronText.split('.')]

        for i in range(len(syllables)):
            syll = Syllable()
            syll.m_language = self.m_language
            syll.m_tone = Tone.parse('0')

            phones = re.split(r'[\s]+', syllables[i])

            for j in range(len(phones)):
                phoneName = phones[j].lower()
                toneName = '0'

                if '0' in phoneName or '1' in phoneName or '2' in phoneName:
                    toneName = phoneName[-1]
                    phoneName = phoneName[:-1]

                phoneName_lst = None
                if self.m_f2t_map is not None:
                    phoneName_lst = self.m_f2t_map.get(phoneName, None)
                if phoneName_lst is None:
                    phoneName_lst = [phoneName]

                for new_phoneName in phoneName_lst:
                    phone_obj = phoneset.m_name_map.get(new_phoneName, None)
                    if phone_obj is None:
                        logging.error(
                            'EnXXSyllableFormatter.Format: phone %s not found',
                            new_phoneName,
                        )
                        return False
                    phone_obj.m_name = new_phoneName
                    syll.m_phone_list.append(phone_obj)
                    if phone_obj.m_cv_type == PhoneCVType.Vowel:
                        syll.m_tone = Tone.parse(toneName)

                    if j == len(phones) - 1:
                        phone_obj.m_bnd = True
            syllable_list.append(syll)
        return True
