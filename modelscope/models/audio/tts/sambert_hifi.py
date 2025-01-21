# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import datetime
import os
import shutil
import wave
import zipfile

import json
import matplotlib.pyplot as plt
import numpy as np
import yaml

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.audio.audio_utils import (TtsCustomParams, TtsTrainType,
                                                ndarray_pcm_to_wav)
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
        self.model_dir = model_dir
        self.sample_rate = kwargs.get('sample_rate', 16000)
        self.is_train = False
        if 'is_train' in kwargs:
            is_train = kwargs['is_train']
            if isinstance(is_train, bool):
                self.is_train = is_train
        # check legacy modelcard
        self.ignore_mask = False
        if 'am' in kwargs:
            if 'linguistic_unit' in kwargs['am']:
                self.ignore_mask = not kwargs['am']['linguistic_unit'].get(
                    'has_mask', True)
        self.voices, self.voice_cfg, self.lang_type = self.load_voice(
            model_dir, kwargs.get('custom_ckpt', {}))
        if len(self.voices) == 0 or len(self.voice_cfg.get('voices', [])) == 0:
            raise TtsVoiceNotExistsException('modelscope error: voices empty')
        if self.voice_cfg['voices']:
            self.default_voice_name = self.voice_cfg['voices'][0]
        else:
            raise TtsVoiceNotExistsException(
                'modelscope error: voices is empty in voices.json')
        # initialize frontend
        import ttsfrd
        frontend = ttsfrd.TtsFrontendEngine()
        zip_file = os.path.join(model_dir, 'resource.zip')
        self.res_path = os.path.join(model_dir, 'resource')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        if not frontend.initialize(self.res_path):
            raise TtsFrontendInitializeFailedException(
                'modelscope error: resource invalid: {}'.format(self.res_path))
        if not frontend.set_lang_type(self.lang_type):
            raise TtsFrontendLanguageTypeInvalidException(
                'modelscope error: language type invalid: {}'.format(
                    self.lang_type))
        self.frontend = frontend

    def build_voice_from_custom(self, model_dir, custom_ckpt):
        necessary_files = (TtsCustomParams.VOICE_NAME, TtsCustomParams.AM_CKPT,
                           TtsCustomParams.VOC_CKPT, TtsCustomParams.AM_CONFIG,
                           TtsCustomParams.VOC_CONFIG)
        voices = {}
        voices_cfg = {}
        lang_type = 'PinYin'
        for k in necessary_files:
            if k not in custom_ckpt:
                raise TtsModelNotExistsException(
                    f'custom ckpt must have: {necessary_files}')
        voice_name = custom_ckpt[TtsCustomParams.VOICE_NAME]
        voice = Voice(
            voice_name=voice_name,
            voice_path=model_dir,
            custom_ckpt=custom_ckpt,
            ignore_mask=self.ignore_mask,
            is_train=self.is_train)
        voices[voice_name] = voice
        voices_cfg['voices'] = [voice_name]
        lang_type = voice.lang_type
        return voices, voices_cfg, lang_type

    def load_voice(self, model_dir, custom_ckpt):
        voices = {}
        voices_path = os.path.join(model_dir, 'voices')
        voices_json_path = os.path.join(voices_path, 'voices.json')
        lang_type = 'PinYin'
        if len(custom_ckpt) != 0:
            return self.build_voice_from_custom(model_dir, custom_ckpt)
        if not os.path.exists(voices_path) or not os.path.exists(
                voices_json_path):
            return voices, {}, lang_type
        with open(voices_json_path, 'r', encoding='utf-8') as f:
            voice_cfg = json.load(f)
        if 'voices' not in voice_cfg:
            return voices, {}, lang_type
        for name in voice_cfg['voices']:
            voice_path = os.path.join(voices_path, name)
            if not os.path.exists(voice_path):
                continue
            voices[name] = Voice(
                name,
                voice_path,
                ignore_mask=self.ignore_mask,
                is_train=self.is_train)
            lang_type = voices[name].lang_type
        return voices, voice_cfg, lang_type

    def save_voices(self):
        voices_json_path = os.path.join(self.model_dir, 'voices',
                                        'voices.json')
        if os.path.exists(voices_json_path):
            os.remove(voices_json_path)
        save_voices = {}
        save_voices['voices'] = []
        for k in self.voices.keys():
            save_voices['voices'].append(k)
        with open(voices_json_path, 'w', encoding='utf-8') as f:
            json.dump(save_voices, f)

    def get_voices(self):
        return self.voices, self.voice_cfg

    def create_empty_voice(self, voice_name, audio_config, am_config_path,
                           voc_config_path):
        voice_name_path = os.path.join(self.model_dir, 'voices', voice_name)
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
        self.voices[voice_name] = Voice(
            voice_name=voice_name,
            voice_path=voice_name_path,
            allow_empty=True)

    def get_voice_audio_config_path(self, voice):
        if voice not in self.voices:
            return ''
        return self.voices[voice].audio_config

    def get_voice_se_model_path(self, voice):
        if voice not in self.voices:
            return ''
        if self.voices[voice].se_enable:
            return self.voices[voice].se_model_path
        else:
            return ''

    def get_voice_lang_path(self, voice):
        if voice not in self.voices:
            return ''
        return self.voices[voice].lang_dir

    def synthesis_one_sentences(self, voice_name, text):
        if voice_name not in self.voices:
            raise TtsVoiceNotExistsException(
                f'modelscope error: Voice {voice_name} not exists')
        return self.voices[voice_name].forward(text)

    def train(self,
              voice,
              dirs,
              train_type,
              configs_path_dict=None,
              ignore_pretrain=False,
              create_if_not_exists=False,
              hparam=None):
        plt.set_loglevel('info')
        work_dir = dirs['work_dir']
        am_dir = dirs['am_tmp_dir']
        voc_dir = dirs['voc_tmp_dir']
        data_dir = dirs['data_dir']
        target_voice = None
        if voice not in self.voices:
            if not create_if_not_exists:
                raise TtsVoiceNotExistsException(
                    f'modelscope error: Voice {voice_name} not exists')
            am_config_path = configs_path_dict.get('am_config',
                                                   'am_config.yaml')
            voc_config_path = configs_path_dict.get('voc_config',
                                                    'voc_config.yaml')
            if TtsTrainType.TRAIN_TYPE_SAMBERT in train_type and not am_config:
                raise TtsTrainingCfgNotExistsException(
                    'training new voice am with empty am_config')
            if TtsTrainType.TRAIN_TYPE_VOC in train_type and not voc_config:
                raise TtsTrainingCfgNotExistsException(
                    'training new voice voc with empty voc_config')
        else:
            target_voice = self.voices[voice]
            am_config_path = target_voice.am_config_path
            voc_config_path = target_voice.voc_config_path
            if configs_path_dict:
                if 'am_config' in configs_path_dict:
                    am_override = configs_path_dict['am_config']
                    if os.path.exists(am_override):
                        am_config_path = am_override
                if 'voc_config' in configs_path_dict:
                    voc_override = configs_path_dict['voc_config']
                    if os.path.exists(voc_override):
                        voc_config_path = voc_override

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
        voice = self.default_voice_name
        if voice_name is not None:
            voice = voice_name
        result = self.frontend.gen_tacotron_symbols(text)
        texts = [s for s in result.splitlines() if s != '']
        audio_total = np.empty((0), dtype='int16')
        for line in texts:
            line = line.strip().split('\t')
            audio = self.synthesis_one_sentences(voice, line[1])
            audio = 32768.0 * audio
            audio_total = np.append(audio_total, audio.astype('int16'), axis=0)
        return ndarray_pcm_to_wav(self.sample_rate, audio_total)
