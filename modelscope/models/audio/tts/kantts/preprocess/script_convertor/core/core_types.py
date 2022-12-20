# Copyright (c) Alibaba, Inc. and its affiliates.

from enum import Enum


class Tone(Enum):
    UnAssigned = -1
    NoneTone = 0
    YinPing = 1  # ZhHK: YinPingYinRu   EnUS: primary stress
    YangPing = 2  # ZhHK: YinShang       EnUS: secondary stress
    ShangSheng = 3  # ZhHK: YinQuZhongRu
    QuSheng = 4  # ZhHK: YangPing
    QingSheng = 5  # ZhHK: YangShang
    YangQuYangRu = 6  # ZhHK: YangQuYangRu

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(Tone, cls).__new__(cls, in_str)

        if in_str in ['UnAssigned', '-1']:
            return Tone.UnAssigned
        elif in_str in ['NoneTone', '0']:
            return Tone.NoneTone
        elif in_str in ['YinPing', '1']:
            return Tone.YinPing
        elif in_str in ['YangPing', '2']:
            return Tone.YangPing
        elif in_str in ['ShangSheng', '3']:
            return Tone.ShangSheng
        elif in_str in ['QuSheng', '4']:
            return Tone.QuSheng
        elif in_str in ['QingSheng', '5']:
            return Tone.QingSheng
        elif in_str in ['YangQuYangRu', '6']:
            return Tone.YangQuYangRu
        else:
            return Tone.NoneTone


class BreakLevel(Enum):
    UnAssigned = -1
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(BreakLevel, cls).__new__(cls, in_str)

        if in_str in ['UnAssigned', '-1']:
            return BreakLevel.UnAssigned
        elif in_str in ['L0', '0']:
            return BreakLevel.L0
        elif in_str in ['L1', '1']:
            return BreakLevel.L1
        elif in_str in ['L2', '2']:
            return BreakLevel.L2
        elif in_str in ['L3', '3']:
            return BreakLevel.L3
        elif in_str in ['L4', '4']:
            return BreakLevel.L4
        else:
            return BreakLevel.UnAssigned


class SentencePurpose(Enum):
    Declarative = 0
    Interrogative = 1
    Exclamatory = 2
    Imperative = 3


class Language(Enum):
    Neutral = 0
    EnUS = 1033
    EnGB = 2057
    ZhCN = 2052
    PinYin = 2053
    WuuShanghai = 2054
    Sichuan = 2055
    ZhHK = 3076
    ZhEn = ZhCN | EnUS

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(Language, cls).__new__(cls, in_str)

        if in_str in ['Neutral', '0']:
            return Language.Neutral
        elif in_str in ['EnUS', '1033']:
            return Language.EnUS
        elif in_str in ['EnGB', '2057']:
            return Language.EnGB
        elif in_str in ['ZhCN', '2052']:
            return Language.ZhCN
        elif in_str in ['PinYin', '2053']:
            return Language.PinYin
        elif in_str in ['WuuShanghai', '2054']:
            return Language.WuuShanghai
        elif in_str in ['Sichuan', '2055']:
            return Language.Sichuan
        elif in_str in ['ZhHK', '3076']:
            return Language.ZhHK
        elif in_str in ['ZhEn', '2052|1033']:
            return Language.ZhEn
        else:
            return Language.Neutral


"""
Phone Types
"""


class PhoneCVType(Enum):
    NULL = -1
    Consonant = 1
    Vowel = 2

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(PhoneCVType, cls).__new__(cls, in_str)

        if in_str in ['consonant', 'Consonant']:
            return PhoneCVType.Consonant
        elif in_str in ['vowel', 'Vowel']:
            return PhoneCVType.Vowel
        else:
            return PhoneCVType.NULL


class PhoneIFType(Enum):
    NULL = -1
    Initial = 1
    Final = 2

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(PhoneIFType, cls).__new__(cls, in_str)
        if in_str in ['initial', 'Initial']:
            return PhoneIFType.Initial
        elif in_str in ['final', 'Final']:
            return PhoneIFType.Final
        else:
            return PhoneIFType.NULL


class PhoneUVType(Enum):
    NULL = -1
    Voiced = 1
    UnVoiced = 2

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(PhoneUVType, cls).__new__(cls, in_str)
        if in_str in ['voiced', 'Voiced']:
            return PhoneUVType.Voiced
        elif in_str in ['unvoiced', 'UnVoiced']:
            return PhoneUVType.UnVoiced
        else:
            return PhoneUVType.NULL


class PhoneAPType(Enum):
    NULL = -1
    DoubleLips = 1
    LipTooth = 2
    FrontTongue = 3
    CentralTongue = 4
    BackTongue = 5
    Dorsal = 6
    Velar = 7
    Low = 8
    Middle = 9
    High = 10

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(PhoneAPType, cls).__new__(cls, in_str)
        if in_str in ['doublelips', 'DoubleLips']:
            return PhoneAPType.DoubleLips
        elif in_str in ['liptooth', 'LipTooth']:
            return PhoneAPType.LipTooth
        elif in_str in ['fronttongue', 'FrontTongue']:
            return PhoneAPType.FrontTongue
        elif in_str in ['centraltongue', 'CentralTongue']:
            return PhoneAPType.CentralTongue
        elif in_str in ['backtongue', 'BackTongue']:
            return PhoneAPType.BackTongue
        elif in_str in ['dorsal', 'Dorsal']:
            return PhoneAPType.Dorsal
        elif in_str in ['velar', 'Velar']:
            return PhoneAPType.Velar
        elif in_str in ['low', 'Low']:
            return PhoneAPType.Low
        elif in_str in ['middle', 'Middle']:
            return PhoneAPType.Middle
        elif in_str in ['high', 'High']:
            return PhoneAPType.High
        else:
            return PhoneAPType.NULL


class PhoneAMType(Enum):
    NULL = -1
    Stop = 1
    Affricate = 2
    Fricative = 3
    Nasal = 4
    Lateral = 5
    Open = 6
    Close = 7

    @classmethod
    def parse(cls, in_str):
        if not isinstance(in_str, str):
            return super(PhoneAMType, cls).__new__(cls, in_str)
        if in_str in ['stop', 'Stop']:
            return PhoneAMType.Stop
        elif in_str in ['affricate', 'Affricate']:
            return PhoneAMType.Affricate
        elif in_str in ['fricative', 'Fricative']:
            return PhoneAMType.Fricative
        elif in_str in ['nasal', 'Nasal']:
            return PhoneAMType.Nasal
        elif in_str in ['lateral', 'Lateral']:
            return PhoneAMType.Lateral
        elif in_str in ['open', 'Open']:
            return PhoneAMType.Open
        elif in_str in ['close', 'Close']:
            return PhoneAMType.Close
        else:
            return PhoneAMType.NULL
