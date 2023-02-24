# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import datetime
import os
import shutil
import wave
import zipfile

import json
import numpy as np
import yaml

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.audio.audio_utils import TtsTrainType, ndarray_pcm_to_wav
from modelscope.utils.audio.tts_exceptions import (
    TtsFrontendInitializeFailedException,
    TtsFrontendLanguageTypeInvalidException, TtsModelConfigurationException,
    TtsVoiceNotExistsException)
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from .voice import Voice

__all__ = ['SambertHifigan']

logger = get_logger()


@MODELS.register_module(
    Tasks.text_to_speech, module_name=Models.sambert_hifigan)
class SambertHifigan(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.__model_dir = model_dir
        self.__sample_rate = kwargs.get('sample_rate', 16000)
        self.__is_train = False
        if 'is_train' in kwargs:
            is_train = kwargs['is_train']
            if isinstance(is_train, bool):
                self.__is_train = is_train
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
        self.__voices, self.__voice_cfg = self.load_voice(model_dir)
        if len(self.__voices) == 0 or len(self.__voice_cfg) == 0:
            raise TtsVoiceNotExistsException('modelscope error: voices empty')
        if self.__voice_cfg['voices']:
            self.__default_voice_name = self.__voice_cfg['voices'][0]
        else:
            raise TtsVoiceNotExistsException(
                'modelscope error: voices is empty in voices.json')

    def load_voice(self, model_dir):
        voices = {}
        voices_path = os.path.join(model_dir, 'voices')
        voices_json_path = os.path.join(voices_path, 'voices.json')
        if not os.path.exists(voices_path) or not os.path.exists(
                voices_json_path):
            return voices, []
        with open(voices_json_path, 'r', encoding='utf-8') as f:
            voice_cfg = json.load(f)
        if 'voices' not in voice_cfg:
            return voices, []
        for name in voice_cfg['voices']:
            voice_path = os.path.join(voices_path, name)
            if not os.path.exists(voice_path):
                continue
            voices[name] = Voice(name, voice_path)
        return voices, voice_cfg

    def save_voices(self):
        voices_json_path = os.path.join(self.__model_dir, 'voices',
                                        'voices.json')
        if os.path.exists(voices_json_path):
            os.remove(voices_json_path)
        save_voices = {}
        save_voices['voices'] = []
        for k in self.__voices.keys():
            save_voices['voices'].append(k)
        with open(voices_json_path, 'w', encoding='utf-8') as f:
            json.dump(save_voices, f)

    def get_voices(self):
        return self.__voices, self.__voice_cfg

    def create_empty_voice(self, voice_name, audio_config, am_config_path,
                           voc_config_path):
        voice_name_path = os.path.join(self.__model_dir, 'voices', voice_name)
        if os.path.exists(voice_name_path):
            shutil.rmtree(voice_name_path)
        os.makedirs(voice_name_path, exist_ok=True)
        if audio_config and os.path.exists(audio_config) and os.path.isfile(
                audio_config):
            shutil.copy(audio_config, voice_name_path)
        voice_am_path = os.path.join(voice_name_path, 'am')
        voice_voc_path = os.path.join(voice_name_path, 'voc')
        if am_config_path and os.path.exists(
                am_config_path) and os.path.isfile(am_config):
            am_config_name = os.path.join(voice_am_path, 'config.yaml')
            shutil.copy(am_config_path, am_config_name)
        if voc_config_path and os.path.exists(
                voc_config_path) and os.path.isfile(voc_config):
            voc_config_name = os.path.join(voice_am_path, 'config.yaml')
            shutil.copy(voc_config_path, voc_config_name)
        am_ckpt_path = os.path.join(voice_am_path, 'ckpt')
        voc_ckpt_path = os.path.join(voice_voc_path, 'ckpt')
        os.makedirs(am_ckpt_path, exist_ok=True)
        os.makedirs(voc_ckpt_path, exist_ok=True)
        self.__voices[voice_name] = Voice(
            voice_name=voice_name,
            voice_path=voice_name_path,
            allow_empty=True)

    def get_voice_audio_config_path(self, voice):
        if voice not in self.__voices:
            return ''
        return self.__voices[voice].audio_config

    def get_voice_lang_path(self, voice):
        if voice not in self.__voices:
            return ''
        return self.__voices[voice].lang_dir

    def __synthesis_one_sentences(self, voice_name, text):
        if voice_name not in self.__voices:
            raise TtsVoiceNotExistsException(
                f'modelscope error: Voice {voice_name} not exists')
        return self.__voices[voice_name].forward(text)

    def train(self,
              voice,
              dirs,
              train_type,
              configs_path=None,
              ignore_pretrain=False,
              create_if_not_exists=False,
              hparam=None):
        work_dir = dirs['work_dir']
        am_dir = dirs['am_tmp_dir']
        voc_dir = dirs['voc_tmp_dir']
        data_dir = dirs['data_dir']

        if voice not in self.__voices:
            if not create_if_not_exists:
                raise TtsVoiceNotExistsException(
                    f'modelscope error: Voice {voice_name} not exists')
            am_config = configs_path.get('am_config', None)
            voc_config = configs_path.get('voc_config', None)
            if TtsTrainType.TRAIN_TYPE_SAMBERT in train_type and not am_config:
                raise TtsTrainingCfgNotExistsException(
                    'training new voice am with empty am_config')
            if TtsTrainType.TRAIN_TYPE_VOC in train_type and not voc_config:
                raise TtsTrainingCfgNotExistsException(
                    'training new voice voc with empty voc_config')

        target_voice = self.__voices[voice]
        am_config_path = target_voice.am_config
        voc_config_path = target_voice.voc_config
        if not configs_path:
            am_config = configs_path.get('am_config', None)
            if am_config:
                am_config_path = am_config
            voc_config = configs_path.get('voc_config', None)
            if voc_config:
                voc_config_path = voc_config

        logger.info('Start training....')
        if TtsTrainType.TRAIN_TYPE_SAMBERT in train_type:
            logger.info('Start SAMBERT training...')
            totaltime = datetime.datetime.now()
            hparams = train_type[TtsTrainType.TRAIN_TYPE_SAMBERT]
            target_voice.train_sambert(work_dir, am_dir, data_dir,
                                       am_config_path, ignore_pretrain,
                                       hparams)
            totaltime = datetime.datetime.now() - totaltime
            logger.info('SAMBERT training spent: {:.2f} hours\n'.format(
                totaltime.total_seconds() / 3600.0))
        else:
            logger.info('skip SAMBERT training...')

        if TtsTrainType.TRAIN_TYPE_VOC in train_type:
            logger.info('Start HIFIGAN training...')
            totaltime = datetime.datetime.now()
            hparams = train_type[TtsTrainType.TRAIN_TYPE_VOC]
            target_voice.train_hifigan(work_dir, voc_dir, data_dir,
                                       voc_config_path, ignore_pretrain,
                                       hparams)
            totaltime = datetime.datetime.now() - totaltime
            logger.info('HIFIGAN training spent: {:.2f} hours\n'.format(
                totaltime.total_seconds() / 3600.0))
        else:
            logger.info('skip HIFIGAN training...')

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
            audio = 32768.0 * audio
            audio_total = np.append(audio_total, audio.astype('int16'), axis=0)
        return ndarray_pcm_to_wav(self.__sample_rate, audio_total)
