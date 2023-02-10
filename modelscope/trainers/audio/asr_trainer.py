# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
from typing import Dict, Optional, Union

import json
from funasr.bin import build_trainer

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import (DEFAULT_DATASET_NAMESPACE,
                                       DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION, ModelFile,
                                       Tasks, TrainerStages)
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.speech_asr_trainer)
class ASRTrainer(BaseTrainer):
    DATA_DIR = 'data'

    def __init__(self,
                 model: str,
                 work_dir: str = None,
                 distributed: bool = False,
                 dataset_type: str = 'small',
                 data_dir: Optional[Union[MsDataset, str]] = None,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 batch_bins: Optional[int] = None,
                 max_epoch: Optional[int] = None,
                 lr: Optional[float] = None,
                 mate_params: Optional[dict] = None,
                 **kwargs):
        """ASR Trainer.

        Args:
            model (str) : model name
            work_dir (str): output dir for saving results
            distributed (bool): whether to enable DDP training
            dataset_type (str): choose which dataset type to use
            data_dir (str): the path of data
            model_revision (str): set model version
            batch_bins (str): batch size
            max_epoch (int): the maximum epoch number for training
            lr (float): learning rate
            mate_params (dict): for saving other training args
        Examples:

        >>> import os
        >>> from modelscope.metainfo import Trainers
        >>> from modelscope.msdatasets import MsDataset
        >>> from modelscope.trainers import build_trainer
        >>> ds_dict = MsDataset.load('speech_asr_aishell1_trainsets')
        >>> kwargs = dict(
        >>>     model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        >>>     data_dir=ds_dict,
        >>>     work_dir="./checkpoint")
        >>> trainer = build_trainer(
        >>>     Trainers.speech_asr_trainer, default_args=kwargs)
        >>> trainer.train()

        """
        if not work_dir:
            self.work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)
        else:
            self.work_dir = work_dir

        if not os.path.exists(self.work_dir):
            raise Exception(f'{self.work_dir} not exists')

        logger.info(f'Set workdir to {self.work_dir}')

        self.data_dir = os.path.join(self.work_dir, self.DATA_DIR)
        self.raw_dataset_path = ''
        self.distributed = distributed
        self.dataset_type = dataset_type

        shutil.rmtree(self.data_dir, ignore_errors=True)

        os.makedirs(self.data_dir, exist_ok=True)

        if os.path.exists(model):
            model_dir = model
        else:
            model_dir = self.get_or_download_model_dir(model, model_revision)
        self.model_dir = model_dir
        self.model_cfg = os.path.join(self.model_dir, 'configuration.json')
        self.cfg_dict = self.parse_cfg(self.model_cfg)

        if 'raw_data_dir' not in data_dir:
            self.train_data_dir, self.dev_data_dir = self.load_dataset_raw_path(
                data_dir, self.data_dir)
        else:
            self.data_dir = data_dir['raw_data_dir']
        self.trainer = build_trainer.build_trainer(
            modelscope_dict=self.cfg_dict,
            data_dir=self.data_dir,
            output_dir=self.work_dir,
            distributed=self.distributed,
            dataset_type=self.dataset_type,
            batch_bins=batch_bins,
            max_epoch=max_epoch,
            lr=lr,
            mate_params=mate_params)

    def parse_cfg(self, cfg_file):
        cur_dir = os.path.dirname(cfg_file)
        cfg_dict = dict()
        with open(cfg_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            cfg_dict['mode'] = config['model']['model_config']['mode']
            cfg_dict['model_dir'] = cur_dir
            cfg_dict['am_model_file'] = os.path.join(
                cur_dir, config['model']['am_model_name'])
            cfg_dict['am_model_config'] = os.path.join(
                cur_dir, config['model']['model_config']['am_model_config'])
            cfg_dict['finetune_config'] = os.path.join(cur_dir,
                                                       'finetune.yaml')
            cfg_dict['cmvn_file'] = os.path.join(
                cur_dir, config['model']['model_config']['mvn_file'])
            cfg_dict['seg_dict'] = os.path.join(cur_dir, 'seg_dict')
            if 'init_model' in config['model']['model_config']:
                cfg_dict['init_model'] = os.path.join(
                    cur_dir, config['model']['model_config']['init_model'])
            else:
                cfg_dict['init_model'] = cfg_dict['am_model_file']
        return cfg_dict

    def load_dataset_raw_path(self, dataset, output_data_dir):
        if 'train' not in dataset:
            raise Exception(
                'dataset {0} does not contain a train split'.format(dataset))
        train_data_dir = self.prepare_data(
            dataset, output_data_dir, split='train')
        if 'validation' not in dataset:
            raise Exception(
                'dataset {0} does not contain a dev split'.format(dataset))
        dev_data_dir = self.prepare_data(
            dataset, output_data_dir, split='validation')
        return train_data_dir, dev_data_dir

    def prepare_data(self, dataset, out_base_dir, split='train'):
        out_dir = os.path.join(out_base_dir, split)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        data_cnt = len(dataset[split])
        fp_wav_scp = open(os.path.join(out_dir, 'wav.scp'), 'w')
        fp_text = open(os.path.join(out_dir, 'text'), 'w')
        for i in range(data_cnt):
            content = dataset[split][i]
            wav_file = content['Audio:FILE']
            text = content['Text:LABEL']
            fp_wav_scp.write('\t'.join([os.path.basename(wav_file), wav_file])
                             + '\n')
            fp_text.write('\t'.join([os.path.basename(wav_file), text]) + '\n')
        fp_text.close()
        fp_wav_scp.close()
        return out_dir

    def train(self, *args, **kwargs):
        self.trainer.run()

    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        raise NotImplementedError
