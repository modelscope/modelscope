# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import json

from modelscope.metainfo import Preprocessors, Trainers
from modelscope.models.audio.tts import SambertHifigan
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.builder import build_preprocessor
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.utils.audio.tts_exceptions import (
    TtsTrainingCfgNotExistsException, TtsTrainingDatasetInvalidException,
    TtsTrainingHparamsInvalidException, TtsTrainingInvalidModelException,
    TtsTrainingWorkDirNotExistsException)
from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION, ModelFile,
                                       Tasks, TrainerStages)
from modelscope.utils.data_utils import to_device
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.speech_kantts_trainer)
class KanttsTrainer(BaseTrainer):
    DATA_DIR = 'data'
    AM_TMP_DIR = 'tmp_am'
    VOC_TMP_DIR = 'tmp_voc'
    ORIG_MODEL_DIR = 'orig_model'

    def __init__(self,
                 model: str,
                 work_dir: str = None,
                 speaker: str = 'F7',
                 lang_type: str = 'PinYin',
                 cfg_file: Optional[str] = None,
                 train_dataset: Optional[Union[MsDataset, str]] = None,
                 train_dataset_namespace: str = DEFAULT_DATASET_NAMESPACE,
                 train_dataset_revision: str = DEFAULT_DATASET_REVISION,
                 train_type: dict = {
                     TtsTrainType.TRAIN_TYPE_SAMBERT: {},
                     TtsTrainType.TRAIN_TYPE_VOC: {}
                 },
                 preprocess_skip_script=False,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 **kwargs):

        if not work_dir:
            self.work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)
        else:
            self.work_dir = work_dir

        if not os.path.exists(self.work_dir):
            raise TtsTrainingWorkDirNotExistsException(
                f'{self.work_dir} not exists')

        self.train_type = dict()
        if isinstance(train_type, dict):
            for k, v in train_type.items():
                if (k == TtsTrainType.TRAIN_TYPE_SAMBERT
                        or k == TtsTrainType.TRAIN_TYPE_VOC
                        or k == TtsTrainType.TRAIN_TYPE_BERT):
                    self.train_type[k] = v

        if len(self.train_type) == 0:
            logger.info('train type empty, default to sambert and voc')
            self.train_type[TtsTrainType.TRAIN_TYPE_SAMBERT] = {}
            self.train_type[TtsTrainType.TRAIN_TYPE_VOC] = {}

        logger.info(f'Set workdir to {self.work_dir}')

        self.data_dir = os.path.join(self.work_dir, self.DATA_DIR)
        self.am_tmp_dir = os.path.join(self.work_dir, self.AM_TMP_DIR)
        self.voc_tmp_dir = os.path.join(self.work_dir, self.VOC_TMP_DIR)
        self.orig_model_dir = os.path.join(self.work_dir, self.ORIG_MODEL_DIR)
        self.raw_dataset_path = ''
        self.skip_script = preprocess_skip_script
        self.audio_config_path = ''
        self.lang_path = ''
        self.am_config_path = ''
        self.voc_config_path = ''

        shutil.rmtree(self.data_dir, ignore_errors=True)
        shutil.rmtree(self.am_tmp_dir, ignore_errors=True)
        shutil.rmtree(self.voc_tmp_dir, ignore_errors=True)
        shutil.rmtree(self.orig_model_dir, ignore_errors=True)

        os.makedirs(self.data_dir)
        os.makedirs(self.am_tmp_dir)
        os.makedirs(self.voc_tmp_dir)

        if train_dataset:
            if isinstance(train_dataset, str):
                logger.info(f'load {train_dataset_namespace}/{train_dataset}')
                train_dataset = MsDataset.load(
                    dataset_name=train_dataset,
                    namespace=train_dataset_namespace,
                    version=train_dataset_revision)
                logger.info(f'train dataset:{train_dataset.config_kwargs}')
            self.raw_dataset_path = self.load_dataset_raw_path(train_dataset)

        model_dir = self.get_or_download_model_dir(model, model_revision)
        shutil.copytree(model_dir, self.orig_model_dir)
        self.model_dir = self.orig_model_dir

        if not cfg_file:
            cfg_file = os.path.join(self.model_dir, ModelFile.CONFIGURATION)
        self.parse_cfg(cfg_file)

        if not os.path.exists(self.raw_dataset_path):
            raise TtsTrainingDatasetInvalidException(
                'dataset raw path not exists')

        self.finetune_from_pretrain = False
        self.speaker = speaker
        self.lang_type = lang_type
        self.model = None
        self.device = kwargs.get('device', 'gpu')
        self.model = self.get_model(self.model_dir, self.speaker,
                                    self.lang_type)
        if TtsTrainType.TRAIN_TYPE_SAMBERT in self.train_type or TtsTrainType.TRAIN_TYPE_VOC in self.train_type:
            self.audio_data_preprocessor = build_preprocessor(
                dict(type=Preprocessors.kantts_data_preprocessor),
                Tasks.text_to_speech)

    def parse_cfg(self, cfg_file):
        cur_dir = os.path.dirname(cfg_file)
        with open(cfg_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            if 'train' not in config:
                raise TtsTrainingInvalidModelException(
                    'model not support finetune')
            if 'audio_config' in config['train']:
                audio_config = os.path.join(cur_dir,
                                            config['train']['audio_config'])
                if os.path.exists(audio_config):
                    self.audio_config_path = audio_config
            if 'am_config' in config['train']:
                am_config = os.path.join(cur_dir, config['train']['am_config'])
                if os.path.exists(am_config):
                    self.am_config_path = am_config
            if 'voc_config' in config['train']:
                voc_config = os.path.join(cur_dir,
                                          config['train']['voc_config'])
                if os.path.exists(voc_config):
                    self.voc_config_path = voc_config
            if 'language_path' in config['train']:
                lang_path = os.path.join(cur_dir,
                                         config['train']['language_path'])
                if os.path.exists(lang_path):
                    self.lang_path = lang_path
            if not self.raw_dataset_path:
                if 'train_dataset' in config['train']:
                    dataset = config['train']['train_dataset']
                    if 'id' in dataset:
                        namespace = dataset.get('namespace',
                                                DEFAULT_DATASET_NAMESPACE)
                        revision = dataset.get('revision',
                                               DEFAULT_DATASET_REVISION)
                        ms = MsDataset.load(
                            dataset_name=dataset['id'],
                            namespace=namespace,
                            version=revision)
                        self.raw_dataset_path = self.load_dataset_raw_path(ms)
                    elif 'path' in dataset:
                        self.raw_dataset_path = dataset['path']

    def load_dataset_raw_path(self, dataset: MsDataset):
        if 'split_config' not in dataset.config_kwargs:
            raise TtsTrainingDatasetInvalidException(
                'split_config not found in config_kwargs')
        if 'train' not in dataset.config_kwargs['split_config']:
            raise TtsTrainingDatasetInvalidException(
                'no train split in split_config')
        return dataset.config_kwargs['split_config']['train']

    def prepare_data(self):
        if self.audio_data_preprocessor:
            audio_config = self.audio_config_path
            if not audio_config or not os.path.exists(audio_config):
                audio_config = self.model.get_voice_audio_config_path(
                    self.speaker)
            lang_path = self.lang_path
            if not lang_path or not os.path.exists(lang_path):
                lang_path = self.model.get_voice_lang_path(self.speaker)
            self.audio_data_preprocessor(self.raw_dataset_path, self.data_dir,
                                         lang_path, audio_config, self.speaker,
                                         self.lang_type, self.skip_script)

    def prepare_text(self):
        pass

    def get_model(self, model_dir, speaker, lang_type):
        model = SambertHifigan(
            model_dir=self.model_dir, lang_type=self.lang_type, is_train=True)
        return model

    def train(self, *args, **kwargs):
        if not self.model:
            raise TtsTrainingInvalidModelException('model is none')
        ignore_pretrain = False
        if 'ignore_pretrain' in kwargs:
            ignore_pretrain = kwargs['ignore_pretrain']

        if TtsTrainType.TRAIN_TYPE_SAMBERT in self.train_type or TtsTrainType.TRAIN_TYPE_VOC in self.train_type:
            self.prepare_data()
        if TtsTrainType.TRAIN_TYPE_BERT in self.train_type:
            self.prepare_text()
        dir_dict = {
            'work_dir': self.work_dir,
            'am_tmp_dir': self.am_tmp_dir,
            'voc_tmp_dir': self.voc_tmp_dir,
            'data_dir': self.data_dir
        }
        config_dict = {
            'am_config': self.am_config_path,
            'voc_config': self.voc_config_path
        }
        self.model.train(self.speaker, dir_dict, self.train_type, config_dict,
                         ignore_pretrain)

    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        return {}
