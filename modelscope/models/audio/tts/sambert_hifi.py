# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import zipfile

import json
import numpy as np

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.audio.tts_exceptions import (
    TtsFrontendInitializeFailedException,
    TtsFrontendLanguageTypeInvalidException, TtsModelConfigurationException,
    TtsVoiceNotExistsException)
from modelscope.utils.constant import Tasks
from .voice import Voice

__all__ = ['SambertHifigan']


@MODELS.register_module(
    Tasks.text_to_speech, module_name=Models.sambert_hifigan)
class SambertHifigan(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        if 'am' not in kwargs:
            raise TtsModelConfigurationException(
                'modelscope error: configuration model field missing am!')
        if 'vocoder' not in kwargs:
            raise TtsModelConfigurationException(
                'modelscope error: configuration model field missing vocoder!')
        if 'lang_type' not in kwargs:
            raise TtsModelConfigurationException(
                'modelscope error: configuration model field missing lang_type!'
            )
        am_cfg = kwargs['am']
        voc_cfg = kwargs['vocoder']
        # initialize frontend
        import ttsfrd
        frontend = ttsfrd.TtsFrontendEngine()
        zip_file = os.path.join(model_dir, 'resource.zip')
        self.__res_path = os.path.join(model_dir, 'resource')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        if not frontend.initialize(self.__res_path):
            raise TtsFrontendInitializeFailedException(
                'modelscope error: resource invalid: {}'.format(
                    self.__res_path))
        if not frontend.set_lang_type(kwargs['lang_type']):
            raise TtsFrontendLanguageTypeInvalidException(
                'modelscope error: language type invalid: {}'.format(
                    kwargs['lang_type']))
        self.__frontend = frontend
        zip_file = os.path.join(model_dir, 'voices.zip')
        self.__voice_path = os.path.join(model_dir, 'voices')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        voice_cfg_path = os.path.join(self.__voice_path, 'voices.json')
        with open(voice_cfg_path, 'r', encoding='utf-8') as f:
            voice_cfg = json.load(f)
        if 'voices' not in voice_cfg:
            raise TtsModelConfigurationException(
                'modelscope error: voices invalid')
        self.__voice = {}
        for name in voice_cfg['voices']:
            voice_path = os.path.join(self.__voice_path, name)
            if not os.path.exists(voice_path):
                continue
            self.__voice[name] = Voice(name, voice_path, am_cfg, voc_cfg)
        if voice_cfg['voices']:
            self.__default_voice_name = voice_cfg['voices'][0]
        else:
            raise TtsVoiceNotExistsException(
                'modelscope error: voices is empty in voices.json')

    def __synthesis_one_sentences(self, voice_name, text):
        if voice_name not in self.__voice:
            raise TtsVoiceNotExistsException(
                f'modelscope error: Voice {voice_name} not exists')
        return self.__voice[voice_name].forward(text)

    def forward(self, text: str, voice_name: str = None):
        voice = self.__default_voice_name
        if voice_name is not None:
            voice = voice_name
        result = self.__frontend.gen_tacotron_symbols(text)
        texts = [s for s in result.splitlines() if s != '']
        audio_total = np.empty((0), dtype='int16')
        for line in texts:
            line = line.strip().split('\t')
            audio = self.__synthesis_one_sentences(voice, line[1])
            audio_total = np.append(audio_total, audio, axis=0)
        return audio_total
