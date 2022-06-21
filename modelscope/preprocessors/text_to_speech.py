# Copyright (c) Alibaba, Inc. and its affiliates.
import io
from typing import Any, Dict, Union

from modelscope.fileio import File
from modelscope.models.audio.tts.frontend import GenericTtsFrontend
from modelscope.models.base import Model
from modelscope.utils.audio.tts_exceptions import *  # noqa F403
from modelscope.utils.constant import Fields
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = ['TextToTacotronSymbols', 'text_to_tacotron_symbols']


@PREPROCESSORS.register_module(
    Fields.audio, module_name=r'text_to_tacotron_symbols')
class TextToTacotronSymbols(Preprocessor):
    """extract tacotron symbols from text.

    Args:
        res_path (str): TTS frontend resource url
        lang_type (str): language type, valid values are "pinyin" and "chenmix"
    """

    def __init__(self, model_name, lang_type='pinyin'):
        self._frontend_model = Model.from_pretrained(
            model_name, lang_type=lang_type)
        assert self._frontend_model is not None, 'load model from pretained failed'

    def __call__(self, data: str) -> Dict[str, Any]:
        """Call functions to load text and get tacotron symbols.

        Args:
            input (str): text with utf-8
        Returns:
            symbos (list[str]): texts in tacotron symbols format.
        """
        return self._frontend_model.forward(data)


def text_to_tacotron_symbols(text='', path='./', lang='pinyin'):
    """ simple interface to transform text to tacotron symbols

    Args:
        text (str): input text
        path (str): resource path
        lang (str): language type from one of "pinyin" and "chenmix"
    """
    transform = TextToTacotronSymbols(path, lang)
    return transform(text)
