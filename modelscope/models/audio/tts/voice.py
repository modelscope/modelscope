# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import pickle as pkl
import time
from collections import OrderedDict
from threading import Lock

import json
import numpy as np
import torch
import yaml
from kantts.datasets.dataset import get_am_datasets, get_voc_datasets
from kantts.models import model_builder
from kantts.train.loss import criterion_builder
from kantts.train.trainer import GAN_Trainer, Sambert_Trainer, distributed_init
from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
from torch.utils.data import DataLoader

from modelscope.utils.audio.audio_utils import TtsCustomParams
from modelscope.utils.audio.tts_exceptions import (
    TtsModelConfigurationException, TtsModelNotExistsException)
from modelscope.utils.logger import get_logger

logger = get_logger()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denorm_f0(mel,
              f0_threshold=30,
              uv_threshold=0.6,
              norm_type='mean_std',
              f0_feature=None):
    if norm_type == 'mean_std':
        f0_mvn = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * f0_mvn[1:, :] + f0_mvn[0:1, :]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv
    else:  # global
        f0_global_max_min = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * (f0_global_max_min[0]
                   - f0_global_max_min[1]) + f0_global_max_min[1]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv

    return mel


def binarize(mel, threshold=0.6):
    # vuv binarize
    res_mel = mel.clone()
    index = torch.where(mel[:, -1] < threshold)[0]
    res_mel[:, -1] = 1.0
    res_mel[:, -1][index] = 0.0
    return res_mel


class Voice:

    def __init__(self,
                 voice_name,
                 voice_path=None,
                 custom_ckpt={},
                 ignore_mask=True,
                 is_train=False):
        self.voice_name = voice_name
        self.voice_path = voice_path
        self.ignore_mask = ignore_mask
        self.is_train = is_train
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            self.distributed = False
        else:
            torch.backends.cudnn.benchmark = True
            self.distributed, self.device, self.local_rank, self.world_size = distributed_init(
            )

        if len(custom_ckpt) != 0:
            self.am_config_path = custom_ckpt[TtsCustomParams.AM_CONFIG]
            self.voc_config_path = custom_ckpt[TtsCustomParams.VOC_CONFIG]
            if not os.path.isabs(self.am_config_path):
                self.am_config_path = os.path.join(voice_path,
                                                   self.am_config_path)
            if not os.path.isabs(self.voc_config_path):
                self.voc_config_path = os.path.join(voice_path,
                                                    self.voc_config_path)
            am_ckpt = custom_ckpt[TtsCustomParams.AM_CKPT]
            voc_ckpt = custom_ckpt[TtsCustomParams.VOC_CKPT]
            if not os.path.isabs(am_ckpt):
                am_ckpt = os.path.join(voice_path, am_ckpt)
            if not os.path.isabs(voc_ckpt):
                voc_ckpt = os.path.join(voice_path, voc_ckpt)
            self.am_ckpts = self.scan_ckpt(am_ckpt)
            self.voc_ckpts = self.scan_ckpt(voc_ckpt)
            self.se_path = custom_ckpt.get(TtsCustomParams.SE_FILE, 'se.npy')
            if not os.path.isabs(self.se_path):
                self.se_path = os.path.join(voice_path, self.se_path)
            self.se_model_path = custom_ckpt.get(TtsCustomParams.SE_MODEL,
                                                 'se.onnx')
            if not os.path.isabs(self.se_model_path):
                self.se_model_path = os.path.join(voice_path,
                                                  self.se_model_path)
            self.audio_config = custom_ckpt.get(TtsCustomParams.AUIDO_CONFIG,
                                                'audio_config.yaml')
            if not os.path.isabs(self.audio_config):
                self.audio_config = os.path.join(voice_path, self.audio_config)
            self.mvn_path = custom_ckpt.get(TtsCustomParams.MVN_FILE,
                                            'mvn.npy')
            if not os.path.isabs(self.mvn_path):
                self.mvn_path = os.path.join(voice_path, self.mvn_path)
        else:
            self.audio_config = os.path.join(voice_path, 'audio_config.yaml')
            self.am_config_path = os.path.join(voice_path, 'am', 'config.yaml')
            self.voc_config_path = os.path.join(voice_path, 'voc',
                                                'config.yaml')

            self.se_path = os.path.join(voice_path, 'am', 'se.npy')
            self.am_ckpts = self.scan_ckpt(
                os.path.join(voice_path, 'am', 'ckpt'))
            self.voc_ckpts = self.scan_ckpt(
                os.path.join(voice_path, 'voc', 'ckpt'))
            self.mvn_path = os.path.join(voice_path, 'am', 'mvn.npy')
            self.se_model_path = os.path.join(voice_path, 'se', 'ckpt',
                                              'se.onnx')

        logger.info(
            f'am_config={self.am_config_path} voc_config={self.voc_config_path}'
        )
        logger.info(f'audio_config={self.audio_config}')
        logger.info(f'am_ckpts={self.am_ckpts}')
        logger.info(f'voc_ckpts={self.voc_ckpts}')
        logger.info(
            f'se_path={self.se_path} se_model_path={self.se_model_path}')
        logger.info(f'mvn_path={self.mvn_path}')

        if not os.path.exists(self.am_config_path):
            raise TtsModelConfigurationException(
                'modelscope error: am configuration not found')
        if not os.path.exists(self.voc_config_path):
            raise TtsModelConfigurationException(
                'modelscope error: voc configuration not found')
        if len(self.am_ckpts) == 0:
            raise TtsModelNotExistsException(
                'modelscope error: am model file not found')
        if len(self.voc_ckpts) == 0:
            raise TtsModelNotExistsException(
                'modelscope error: voc model file not found')
        with open(self.am_config_path, 'r') as f:
            self.am_config = yaml.load(f, Loader=yaml.Loader)
        with open(self.voc_config_path, 'r') as f:
            self.voc_config = yaml.load(f, Loader=yaml.Loader)
        if 'linguistic_unit' not in self.am_config:
            raise TtsModelConfigurationException(
                'no linguistic_unit in am config')
        self.lang_type = self.am_config['linguistic_unit'].get(
            'language', 'PinYin')
        self.model_loaded = False
        self.lock = Lock()
        self.ling_unit = KanTtsLinguisticUnit(self.am_config)
        self.ling_unit_size = self.ling_unit.get_unit_size()
        if self.ignore_mask:
            target_set = set(('sy', 'tone', 'syllable_flag', 'word_segment',
                              'emotion', 'speaker'))
            for k, v in self.ling_unit_size.items():
                if k in target_set:
                    self.ling_unit_size[k] = v - 1

        self.am_config['Model']['KanTtsSAMBERT']['params'].update(
            self.ling_unit_size)

        self.se_enable = self.am_config['Model']['KanTtsSAMBERT'][
            'params'].get('SE', False)
        if self.se_enable and not self.is_train:
            if not os.path.exists(self.se_path):
                raise TtsModelConfigurationException(
                    f'se enabled but se_file:{self.se_path} not exists')
            self.se = np.load(self.se_path)

        self.nsf_enable = self.am_config['Model']['KanTtsSAMBERT'][
            'params'].get('NSF', False)
        if self.nsf_enable and not self.is_train:
            self.nsf_norm_type = self.am_config['Model']['KanTtsSAMBERT'][
                'params'].get('nsf_norm_type', 'mean_std')
            if self.nsf_norm_type == 'mean_std':
                if not os.path.exists(self.mvn_path):
                    raise TtsModelNotExistsException(
                        f'f0_mvn_file: {self.mvn_path} not exists')
                self.f0_feature = np.load(self.mvn_path)
            else:  # global
                nsf_f0_global_minimum = self.am_config['Model'][
                    'KanTtsSAMBERT']['params'].get('nsf_f0_global_minimum',
                                                   30.0)
                nsf_f0_global_maximum = self.am_config['Model'][
                    'KanTtsSAMBERT']['params'].get('nsf_f0_global_maximum',
                                                   730.0)
                self.f0_feature = [
                    nsf_f0_global_maximum, nsf_f0_global_minimum
                ]

    def scan_ckpt(self, ckpt_path):
        select_target = ckpt_path
        input_not_dir = False
        if not os.path.isdir(ckpt_path):
            input_not_dir = True
            ckpt_path = os.path.dirname(ckpt_path)
        filelist = os.listdir(ckpt_path)
        if len(filelist) == 0:
            return {}
        ckpts = {}
        for filename in filelist:
            # checkpoint_X.pth
            if len(filename) - 15 <= 0:
                continue
            if filename[-4:] == '.pth' and filename[0:10] == 'checkpoint':
                filename_prefix = filename.split('.')[0]
                idx = int(filename_prefix.split('_')[-1])
                path = os.path.join(ckpt_path, filename)
                if input_not_dir and path != select_target:
                    continue
                ckpts[idx] = path
        od = OrderedDict(sorted(ckpts.items()))
        return od

    def load_am(self):
        self.am_model, _, _ = model_builder(self.am_config, self.device)
        self.am = self.am_model['KanTtsSAMBERT']
        state_dict = torch.load(
            self.am_ckpts[next(reversed(self.am_ckpts))],
            map_location=self.device)
        self.am.load_state_dict(state_dict['model'], strict=False)
        self.am.eval()

    def load_vocoder(self):
        from kantts.models.hifigan.hifigan import Generator
        self.voc_model = Generator(
            **self.voc_config['Model']['Generator']['params'])
        states = torch.load(
            self.voc_ckpts[next(reversed(self.voc_ckpts))],
            map_location=self.device)
        self.voc_model.load_state_dict(states['model']['generator'])
        if self.voc_config['Model']['Generator']['params']['out_channels'] > 1:
            from kantts.models.pqmf import PQMF
            self.voc_model = PQMF()
        self.voc_model.remove_weight_norm()
        self.voc_model.eval().to(self.device)

    def am_forward(self, symbol_seq):
        with self.lock:
            with torch.no_grad():
                inputs_feat_lst = self.ling_unit.encode_symbol_sequence(
                    symbol_seq)
                inputs_feat_index = 0
                if self.ling_unit.using_byte():
                    inputs_byte_index = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.device))
                    inputs_ling = torch.stack([inputs_byte_index],
                                              dim=-1).unsqueeze(0)
                else:
                    inputs_sy = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.device))
                    inputs_feat_index = inputs_feat_index + 1
                    inputs_tone = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.device))
                    inputs_feat_index = inputs_feat_index + 1
                    inputs_syllable = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.device))
                    inputs_feat_index = inputs_feat_index + 1
                    inputs_ws = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.device))
                    inputs_ling = torch.stack(
                        [inputs_sy, inputs_tone, inputs_syllable, inputs_ws],
                        dim=-1).unsqueeze(0)
                inputs_feat_index = inputs_feat_index + 1
                inputs_emo = (
                    torch.from_numpy(
                        inputs_feat_lst[inputs_feat_index]).long().to(
                            self.device).unsqueeze(0))
                inputs_feat_index = inputs_feat_index + 1
                if self.se_enable:
                    inputs_spk = (
                        torch.from_numpy(
                            self.se.repeat(
                                len(inputs_feat_lst[inputs_feat_index]),
                                axis=0)).float().to(
                                    self.device).unsqueeze(0)[:, :-1, :])
                else:
                    inputs_spk = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.device).unsqueeze(0)[:, :-1])
                inputs_len = (torch.zeros(1).to(self.device).long()
                              + inputs_emo.size(1) - 1)  # minus 1 for "~"
                res = self.am(inputs_ling[:, :-1, :], inputs_emo[:, :-1],
                              inputs_spk, inputs_len)
                postnet_outputs = res['postnet_outputs']
                LR_length_rounded = res['LR_length_rounded']
                valid_length = int(LR_length_rounded[0].item())
                mel_post = postnet_outputs[0, :valid_length, :].cpu()
                if self.nsf_enable:
                    mel_post = denorm_f0(
                        mel_post,
                        norm_type=self.nsf_norm_type,
                        f0_feature=self.f0_feature)
                return mel_post

    def vocoder_forward(self, melspec):
        with torch.no_grad():
            x = melspec.to(self.device)
            if self.voc_model.nsf_enable:
                x = binarize(x)
            x = x.transpose(1, 0).unsqueeze(0)
            y = self.voc_model(x)
            if hasattr(self.voc_model, 'pqmf'):
                y = self.voc_model.synthesis(y)
            y = y.view(-1).cpu().numpy()
            return y

    def train_sambert(self,
                      work_dir,
                      stage_dir,
                      data_dir,
                      config_path,
                      ignore_pretrain=False,
                      hparams=dict()):
        logger.info('TRAIN SAMBERT....')
        if len(self.am_ckpts) == 0:
            raise TtsTrainingInvalidModelException(
                'resume pretrain but model is empty')

        from_steps = hparams.get('resume_from_steps', -1)
        if from_steps < 0:
            from_latest = hparams.get('resume_from_latest', True)
        else:
            from_latest = hparams.get('resume_from_latest', False)
        train_steps = hparams.get('train_steps', 0)

        with open(self.audio_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        with open(config_path, 'r') as f:
            config.update(yaml.load(f, Loader=yaml.Loader))
            config.update(hparams)

        resume_from = None
        if from_latest:
            from_steps = next(reversed(self.am_ckpts))
            resume_from = self.am_ckpts[from_steps]
            if not os.path.exists(resume_from):
                raise TtsTrainingInvalidModelException(
                    f'latest model:{resume_from} not exists')
        else:
            if from_steps not in self.am_ckpts:
                raise TtsTrainingInvalidModelException(
                    f'no such model from steps:{from_steps}')
            else:
                resume_from = self.am_ckpts[from_steps]

        if train_steps > 0:
            train_max_steps = train_steps + from_steps
            config['train_max_steps'] = train_max_steps

        logger.info(f'TRAINING steps: {train_max_steps}')
        config['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime())
        from modelscope import __version__
        config['modelscope_version'] = __version__

        with open(os.path.join(stage_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

        for key, value in config.items():
            logger.info(f'{key} = {value}')

        if self.distributed:
            config['rank'] = torch.distributed.get_rank()
            config['distributed'] = True

        if self.se_enable:
            valid_enable = False
            valid_split_ratio = 0.00
        else:
            valid_enable = True
            valid_split_ratio = 0.02

        fp_enable = config['Model']['KanTtsSAMBERT']['params'].get('FP', False)
        meta_file = [
            os.path.join(
                d,
                'raw_metafile.txt' if not fp_enable else 'fprm_metafile.txt')
            for d in data_dir
        ]

        train_dataset, valid_dataset = get_am_datasets(
            meta_file,
            data_dir,
            config,
            config['allow_cache'],
            split_ratio=1.0 - valid_split_ratio)

        logger.info(f'The number of training files = {len(train_dataset)}.')
        logger.info(f'The number of validation files = {len(valid_dataset)}.')

        sampler = {'train': None, 'valid': None}

        if self.distributed:
            # setup sampler for distributed training
            from torch.utils.data.distributed import DistributedSampler

            sampler['train'] = DistributedSampler(
                dataset=train_dataset,
                num_replicas=self.world_size,
                shuffle=True,
            )
            sampler['valid'] = DistributedSampler(
                dataset=valid_dataset,
                num_replicas=self.world_size,
                shuffle=False,
            ) if valid_enable else None

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=False if self.distributed else True,
            collate_fn=train_dataset.collate_fn,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            sampler=sampler['train'],
            pin_memory=config['pin_memory'],
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            shuffle=False if self.distributed else True,
            collate_fn=valid_dataset.collate_fn,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            sampler=sampler['valid'],
            pin_memory=config['pin_memory'],
        ) if valid_enable else None

        ling_unit_size = train_dataset.ling_unit.get_unit_size()

        config['Model']['KanTtsSAMBERT']['params'].update(ling_unit_size)
        model, optimizer, scheduler = model_builder(config, self.device,
                                                    self.local_rank,
                                                    self.distributed)

        criterion = criterion_builder(config, self.device)

        trainer = Sambert_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.device,
            sampler=sampler,
            train_loader=train_dataloader,
            valid_loader=valid_dataloader,
            max_steps=train_max_steps,
            save_dir=stage_dir,
            save_interval=config['save_interval_steps'],
            valid_interval=config['eval_interval_steps'],
            log_interval=config['log_interval'],
            grad_clip=config['grad_norm'],
        )

        if resume_from is not None:
            trainer.load_checkpoint(resume_from, True, True)
            logger.info(f'Successfully resumed from {resume_from}.')

        try:
            trainer.train()
        except (Exception, KeyboardInterrupt) as e:
            logger.error(e, exc_info=True)
            trainer.save_checkpoint(
                os.path.join(
                    os.path.join(stage_dir, 'ckpt'),
                    f'checkpoint-{trainer.steps}.pth'))
            logger.info(
                f'Successfully saved checkpoint @ {trainer.steps}steps.')

    def train_hifigan(self,
                      work_dir,
                      stage_dir,
                      data_dir,
                      config_path,
                      ignore_pretrain=False,
                      hparams=dict()):
        logger.info('TRAIN HIFIGAN....')
        if len(self.voc_ckpts) == 0:
            raise TtsTrainingInvalidModelException(
                'resume pretrain but model is empty')

        from_steps = hparams.get('resume_from_steps', -1)
        if from_steps < 0:
            from_latest = hparams.get('resume_from_latest', True)
        else:
            from_latest = hparams.get('resume_from_latest', False)
        train_steps = hparams.get('train_steps', 0)

        with open(self.audio_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        with open(config_path, 'r') as f:
            config.update(yaml.load(f, Loader=yaml.Loader))
            config.update(hparams)

        resume_from = None
        if from_latest:
            from_steps = next(reversed(self.voc_ckpts))
            resume_from = self.voc_ckpts[from_steps]
            if not os.path.exists(resume_from):
                raise TtsTrainingInvalidModelException(
                    f'latest model:{resume_from} not exists')
        else:
            if from_steps not in self.voc_ckpts:
                raise TtsTrainingInvalidModelException(
                    f'no such model from steps:{from_steps}')
            else:
                resume_from = self.voc_ckpts[from_steps]

        if train_steps > 0:
            train_max_steps = train_steps
            config['train_max_steps'] = train_max_steps

        logger.info(f'TRAINING steps: {train_max_steps}')
        logger.info(f'resume from: {resume_from}')
        config['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime())
        from modelscope import __version__
        config['modelscope_version'] = __version__

        with open(os.path.join(stage_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

        for key, value in config.items():
            logger.info(f'{key} = {value}')

        train_dataset, valid_dataset = get_voc_datasets(config, data_dir)

        logger.info(f'The number of training files = {len(train_dataset)}.')
        logger.info(f'The number of validation files = {len(valid_dataset)}.')

        sampler = {'train': None, 'valid': None}
        if self.distributed:
            # setup sampler for distributed training
            from torch.utils.data.distributed import DistributedSampler

            sampler['train'] = DistributedSampler(
                dataset=train_dataset,
                num_replicas=self.world_size,
                shuffle=True,
            )
            sampler['valid'] = DistributedSampler(
                dataset=valid_dataset,
                num_replicas=self.world_size,
                shuffle=False,
            )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=False if self.distributed else True,
            collate_fn=train_dataset.collate_fn,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            sampler=sampler['train'],
            pin_memory=config['pin_memory'],
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            shuffle=False if self.distributed else True,
            collate_fn=valid_dataset.collate_fn,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            sampler=sampler['valid'],
            pin_memory=config['pin_memory'],
        )

        model, optimizer, scheduler = model_builder(config, self.device,
                                                    self.local_rank,
                                                    self.distributed)

        criterion = criterion_builder(config, self.device)
        trainer = GAN_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.device,
            sampler=sampler,
            train_loader=train_dataloader,
            valid_loader=valid_dataloader,
            max_steps=train_max_steps,
            save_dir=stage_dir,
            save_interval=config['save_interval_steps'],
            valid_interval=config['eval_interval_steps'],
            log_interval=config['log_interval_steps'],
        )

        if resume_from is not None:
            trainer.load_checkpoint(resume_from)
            logger.info(f'Successfully resumed from {resume_from}.')

        try:
            trainer.train()
        except (Exception, KeyboardInterrupt) as e:
            logger.error(e, exc_info=True)
            trainer.save_checkpoint(
                os.path.join(
                    os.path.join(stage_dir, 'ckpt'),
                    f'checkpoint-{trainer.steps}.pth'))
            logger.info(
                f'Successfully saved checkpoint @ {trainer.steps}steps.')

    def forward(self, symbol_seq):
        with self.lock:
            if not self.model_loaded:
                self.load_am()
                self.load_vocoder()
                self.model_loaded = True
        return self.vocoder_forward(self.am_forward(symbol_seq))
