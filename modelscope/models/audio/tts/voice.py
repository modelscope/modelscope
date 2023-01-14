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
from torch.utils.data import DataLoader

from modelscope import __version__
from modelscope.utils.audio.tts_exceptions import (
    TtsModelConfigurationException, TtsModelNotExistsException)
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

from modelscope.models.audio.tts.kantts import (  # isort:skip; isort:skip
    GAN_Trainer, Generator, KanTtsLinguisticUnit, Sambert_Trainer,
    criterion_builder, get_am_datasets, get_voc_datasets, model_builder)

logger = get_logger()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Voice:

    def __init__(self, voice_name, voice_path, allow_empty=False):
        self.__voice_name = voice_name
        self.__voice_path = voice_path
        self.distributed = False
        self.local_rank = 0
        am_config_path = os.path.join(
            os.path.join(voice_path, 'am'), 'config.yaml')
        voc_config_path = os.path.join(
            os.path.join(voice_path, 'voc'), 'config.yaml')

        self.audio_config = os.path.join(voice_path, 'audio_config.yaml')
        self.lang_dir = os.path.join(voice_path, 'dict')
        self.am_config = am_config_path
        self.voc_config = voc_config_path

        am_ckpt = os.path.join(os.path.join(voice_path, 'am'), 'ckpt')
        voc_ckpt = os.path.join(os.path.join(voice_path, 'voc'), 'ckpt')

        self.__am_ckpts = self.scan_ckpt(am_ckpt)
        self.__voc_ckpts = self.scan_ckpt(voc_ckpt)

        if not os.path.exists(am_config_path):
            raise TtsModelConfigurationException(
                'modelscope error: am configuration not found')
        if not os.path.exists(voc_config_path):
            raise TtsModelConfigurationException(
                'modelscope error: voc configuration not found')
        if not allow_empty:
            if len(self.__am_ckpts) == 0:
                raise TtsModelNotExistsException(
                    'modelscope error: am model file not found')
            if len(self.__voc_ckpts) == 0:
                raise TtsModelNotExistsException(
                    'modelscope error: voc model file not found')
        with open(am_config_path, 'r') as f:
            self.__am_config = yaml.load(f, Loader=yaml.Loader)
        with open(voc_config_path, 'r') as f:
            self.__voc_config = yaml.load(f, Loader=yaml.Loader)
        self.__model_loaded = False
        self.__lock = Lock()
        self.__ling_unit = KanTtsLinguisticUnit(self.__am_config,
                                                self.lang_dir)
        self.__ling_unit_size = self.__ling_unit.get_unit_size()
        self.__am_config['Model']['KanTtsSAMBERT']['params'].update(
            self.__ling_unit_size)
        if torch.cuda.is_available():
            self.__device = torch.device('cuda')
        else:
            self.__device = torch.device('cpu')

    def scan_ckpt(self, ckpt_path):
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
                ckpts[idx] = path
        od = OrderedDict(sorted(ckpts.items()))
        return od

    def __load_am(self):
        self.__am_model, _, _ = model_builder(self.__am_config, self.__device)
        self.__am = self.__am_model['KanTtsSAMBERT']
        state_dict = torch.load(self.__am_ckpts[next(
            reversed(self.__am_ckpts))])
        self.__am.load_state_dict(state_dict['model'], strict=False)
        self.__am.eval()

    def __load_vocoder(self):
        self.__voc_model = Generator(
            **self.__voc_config['Model']['Generator']['params'])
        states = torch.load(self.__voc_ckpts[next(reversed(self.__voc_ckpts))])
        self.__voc_model.load_state_dict(states['model']['generator'])
        if self.__voc_config['Model']['Generator']['params'][
                'out_channels'] > 1:
            from .kantts.models.pqmf import PQMF
            self.__voc_model = PQMF()
        self.__voc_model.remove_weight_norm()
        self.__voc_model.eval().to(self.__device)

    def __am_forward(self, symbol_seq):
        with self.__lock:
            with torch.no_grad():
                inputs_feat_lst = self.__ling_unit.encode_symbol_sequence(
                    symbol_seq)
                inputs_feat_index = 0
                if self.__ling_unit.using_byte():
                    inputs_byte_index = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.__device))
                    inputs_ling = torch.stack([inputs_byte_index],
                                              dim=-1).unsqueeze(0)
                else:
                    inputs_sy = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.__device))
                    inputs_feat_index = inputs_feat_index + 1
                    inputs_tone = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.__device))
                    inputs_feat_index = inputs_feat_index + 1
                    inputs_syllable = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.__device))
                    inputs_feat_index = inputs_feat_index + 1
                    inputs_ws = (
                        torch.from_numpy(
                            inputs_feat_lst[inputs_feat_index]).long().to(
                                self.__device))
                    inputs_ling = torch.stack(
                        [inputs_sy, inputs_tone, inputs_syllable, inputs_ws],
                        dim=-1).unsqueeze(0)
                inputs_feat_index = inputs_feat_index + 1
                inputs_emo = (
                    torch.from_numpy(
                        inputs_feat_lst[inputs_feat_index]).long().to(
                            self.__device).unsqueeze(0))
                inputs_feat_index = inputs_feat_index + 1
                inputs_spk = (
                    torch.from_numpy(
                        inputs_feat_lst[inputs_feat_index]).long().to(
                            self.__device).unsqueeze(0))
                inputs_len = (torch.zeros(1).to(self.__device).long()
                              + inputs_emo.size(1) - 1)  # minus 1 for "~"
                res = self.__am(inputs_ling[:, :-1, :], inputs_emo[:, :-1],
                                inputs_spk[:, :-1], inputs_len)
                postnet_outputs = res['postnet_outputs']
                LR_length_rounded = res['LR_length_rounded']
                valid_length = int(LR_length_rounded[0].item())
                postnet_outputs = postnet_outputs[0, :valid_length, :].cpu()
                return postnet_outputs

    def __binarize(mel, threshold=0.6):
        # vuv binarize
        res_mel = mel.clone()
        index = torch.where(mel[:, -1] < threshold)[0]
        res_mel[:, -1] = 1.0
        res_mel[:, -1][index] = 0.0
        return res_mel

    def __vocoder_forward(self, melspec):
        with torch.no_grad():
            x = melspec.to(self.__device)
            if self.__voc_model.nsf_enable:
                x = self.__binarize(x)
            x = x.transpose(1, 0).unsqueeze(0)
            y = self.__voc_model(x)
            if hasattr(self.__voc_model, 'pqmf'):
                y = self.__voc_model.synthesis(y)
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
        if len(self.__am_ckpts) == 0:
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
            from_steps = next(reversed(self.__am_ckpts))
            resume_from = self.__am_ckpts[from_steps]
            if not os.path.exists(resume_from):
                raise TtsTrainingInvalidModelException(
                    f'latest model:{resume_from} not exists')
        else:
            if from_steps not in self.__am_ckpts:
                raise TtsTrainingInvalidModelException(
                    f'no such model from steps:{from_steps}')
            else:
                resume_from = self.__am_ckpts[from_steps]

        if train_steps > 0:
            train_max_steps = train_steps + from_steps
            config['train_max_steps'] = train_max_steps

        logger.info(f'TRAINING steps: {train_max_steps}')
        config['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime())
        config['modelscope_version'] = __version__

        with open(os.path.join(stage_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

        for key, value in config.items():
            logger.info(f'{key} = {value}')

        fp_enable = config['Model']['KanTtsSAMBERT']['params'].get('FP', False)
        meta_file = [
            os.path.join(
                d,
                'raw_metafile.txt' if not fp_enable else 'fprm_metafile.txt')
            for d in data_dir
        ]

        train_dataset, valid_dataset = get_am_datasets(meta_file, data_dir,
                                                       self.lang_dir, config,
                                                       config['allow_cache'])

        logger.info(f'The number of training files = {len(train_dataset)}.')
        logger.info(f'The number of validation files = {len(valid_dataset)}.')

        sampler = {'train': None, 'valid': None}

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

        ling_unit_size = train_dataset.ling_unit.get_unit_size()

        config['Model']['KanTtsSAMBERT']['params'].update(ling_unit_size)
        model, optimizer, scheduler = model_builder(config, self.__device,
                                                    self.local_rank,
                                                    self.distributed)

        criterion = criterion_builder(config, self.__device)

        trainer = Sambert_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.__device,
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
        if len(self.__voc_ckpts) == 0:
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
            from_steps = next(reversed(self.__voc_ckpts))
            resume_from = self.__voc_ckpts[from_steps]
            if not os.path.exists(resume_from):
                raise TtsTrainingInvalidModelException(
                    f'latest model:{resume_from} not exists')
        else:
            if from_steps not in self.__voc_ckpts:
                raise TtsTrainingInvalidModelException(
                    f'no such model from steps:{from_steps}')
            else:
                resume_from = self.__voc_ckpts[from_steps]

        if train_steps > 0:
            train_max_steps = train_steps
            config['train_max_steps'] = train_max_steps

        logger.info(f'TRAINING steps: {train_max_steps}')
        logger.info(f'resume from: {resume_from}')
        config['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime())
        config['modelscope_version'] = __version__

        with open(os.path.join(stage_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

        for key, value in config.items():
            logger.info(f'{key} = {value}')

        train_dataset, valid_dataset = get_voc_datasets(config, data_dir)

        logger.info(f'The number of training files = {len(train_dataset)}.')
        logger.info(f'The number of validation files = {len(valid_dataset)}.')

        sampler = {'train': None, 'valid': None}

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

        model, optimizer, scheduler = model_builder(config, self.__device,
                                                    self.local_rank,
                                                    self.distributed)

        criterion = criterion_builder(config, self.__device)
        trainer = GAN_Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.__device,
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
        with self.__lock:
            if not self.__model_loaded:
                self.__load_am()
                self.__load_vocoder()
                self.__model_loaded = True
        return self.__vocoder_forward(self.__am_forward(symbol_seq))
