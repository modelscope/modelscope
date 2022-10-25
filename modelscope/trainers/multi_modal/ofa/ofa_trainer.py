# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
import shutil
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import Dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ConfigKeys,
                                       ModeKeys)
from .ofa_trainer_utils import (AdjustLabelSmoothedCrossEntropyCriterion,
                                get_schedule)


@TRAINERS.register_module(module_name=Trainers.ofa)
class OFATrainer(EpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Union[Callable, Dict[str,
                                                         Callable]]] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Union[Preprocessor,
                                         Dict[str, Preprocessor]]] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            seed: int = 42,
            **kwargs):
        model = Model.from_pretrained(model, revision=model_revision)
        model_dir = model.model_dir
        cfg = Config.from_file(cfg_file)
        if 'work_dir' not in kwargs or len(kwargs['work_dir']) == 0:
            work_dir = cfg.train.work_dir
        else:
            work_dir = kwargs['work_dir']
        tokenizer_files = {
            'zh': [
                'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
                'config.json'
            ],
            'en':
            ['tokenizer.json', 'vocab.json', 'merges.txt', 'config.json'],
        }
        for filename in tokenizer_files[cfg.model.get('language', 'en')]:
            finetune_file = os.path.join(work_dir, filename)
            pretrain_file = os.path.join(model_dir, filename)
            if os.path.exists(finetune_file):
                continue
            if os.path.exists(pretrain_file):
                shutil.copy(pretrain_file, finetune_file)

        if preprocessor is None:
            preprocessor = {
                ConfigKeys.train:
                OfaPreprocessor(
                    model_dir=work_dir, mode=ModeKeys.TRAIN, no_collate=True),
                ConfigKeys.val:
                OfaPreprocessor(
                    model_dir=work_dir, mode=ModeKeys.EVAL, no_collate=True),
            }
        # use torchrun launch
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        epoch_steps = math.ceil(
            len(train_dataset) /  # noqa
            (cfg.train.dataloader.batch_size_per_gpu * world_size))  # noqa
        cfg.train.lr_scheduler.num_train_steps = epoch_steps * cfg.train.max_epochs
        cfg.train.criterion.tokenizer = model.tokenizer
        self.criterion = AdjustLabelSmoothedCrossEntropyCriterion(
            cfg.train.criterion)
        if optimizers[0] is None:
            optimizer = build_optimizer(model, cfg=cfg.train.optimizer)
        else:
            optimizer = optimizers[0]
        if optimizers[1] is None:
            scheduler_class, scheduler_args = get_schedule(
                cfg.train.lr_scheduler)
            if scheduler_class is not None:
                lr_scheduler = scheduler_class(**{'optimizer': optimizer},
                                               **scheduler_args)
            else:
                lr_scheduler = None
        else:
            lr_scheduler = optimizers[1]
        optimizers = (optimizer, lr_scheduler)
        if data_collator is None:
            data_collator = partial(
                collate_fn,
                pad_idx=model.tokenizer.pad_token_id,
                eos_idx=model.tokenizer.eos_token_id,
            )
        if 'launcher' not in kwargs and cfg.train.get('launcher', None):
            kwargs['launcher'] = cfg.train.launcher
        if 'use_fp16' not in kwargs and cfg.train.get('use_fp16', False):
            kwargs['use_fp16'] = cfg.train.use_fp16
        kwargs['to_tensor'] = False
        super().__init__(
            model=model,
            cfg_file=cfg_file,
            arg_parse_fn=arg_parse_fn,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessor=preprocessor,
            optimizers=optimizers,
            seed=seed,
            **kwargs,
        )

    def train_step(self, model, inputs):
        model.train()
        # model_outputs = model.forward(inputs)
        loss, sample_size, logging_output = self.criterion(model, inputs)
        train_outputs = {'loss': loss}
        # add model output info to log
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update(
                    [key for key in train_outputs.keys() if key_p in key])
            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if dist.is_available() and dist.is_initialized():
                        value = value.data.clone()
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])
        self.train_outputs = train_outputs
