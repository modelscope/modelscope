import os
import zipfile
from typing import Any, Dict, List

from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.audio.tts_exceptions import (
    TtsFrontendInitializeFailedException,
    TtsFrontendLanguageTypeInvalidException)
from modelscope.utils.constant import Tasks

__all__ = ['GenericTtsFrontend']


@MODELS.register_module(
    Tasks.text_to_speech, module_name=r'generic_tts_frontend')
class GenericTtsFrontend(Model):

    def __init__(self, model_dir='.', lang_type='pinyin', *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        import ttsfrd

        frontend = ttsfrd.TtsFrontendEngine()
        zip_file = os.path.join(model_dir, 'resource.zip')
        self._res_path = os.path.join(model_dir, 'resource')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        if not frontend.initialize(self._res_path):
            raise TtsFrontendInitializeFailedException(
                'resource invalid: {}'.format(self._res_path))
        if not frontend.set_lang_type(lang_type):
            raise TtsFrontendLanguageTypeInvalidException(
                'language type invalid: {}, valid is pinyin and chenmix'.
                format(lang_type))
        self._frontend = frontend

    def forward(self, data: str) -> Dict[str, List]:
        result = self._frontend.gen_tacotron_symbols(data)
        return {'texts': [s for s in result.splitlines() if s != '']}
