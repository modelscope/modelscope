# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.constant import Fields, Requirements
from modelscope.utils.import_utils import requires


def get_msg(field):
    msg = f'\n{field} requirements not installed, please execute ' \
          f'`pip install requirements/{field}.txt` or ' \
          f'`pip install modelscope[{field}]`'
    return msg


class NLPModuleNotFoundError(ModuleNotFoundError):

    def __init__(self, e: ModuleNotFoundError) -> None:
        e.msg += get_msg(Fields.nlp)
        super().__init__(e)


class CVModuleNotFoundError(ModuleNotFoundError):

    def __init__(self, e: ModuleNotFoundError) -> None:
        e.msg += get_msg(Fields.cv)
        super().__init__(e)


class AudioModuleNotFoundError(ModuleNotFoundError):

    def __init__(self, e: ModuleNotFoundError) -> None:
        e.msg += get_msg(Fields.audio)
        super().__init__(e)


class MultiModalModuleNotFoundError(ModuleNotFoundError):

    def __init__(self, e: ModuleNotFoundError) -> None:
        e.msg += get_msg(Fields.multi_modal)
        super().__init__(e)


def check_nlp():
    try:
        requires('nlp models', (
            Requirements.torch,
            Requirements.tokenizers,
        ))
    except ImportError as e:
        raise NLPModuleNotFoundError(e)


def check_cv():
    try:
        requires('cv models', (
            Requirements.torch,
            Requirements.tokenizers,
        ))
    except ImportError as e:
        raise CVModuleNotFoundError(e)


def check_audio():
    try:
        requires('audio models', (
            Requirements.torch,
            Requirements.tf,
        ))
    except ImportError as e:
        raise AudioModuleNotFoundError(e)


def check_multi_modal():
    try:
        requires('multi-modal models', (
            Requirements.torch,
            Requirements.tokenizers,
        ))
    except ImportError as e:
        raise MultiModalModuleNotFoundError(e)
