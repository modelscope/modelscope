# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
from functools import partial

from datasets import load_dataset
from torch import distributed as dist

from modelscope.metainfo import Trainers
from modelscope.models.base import Model
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config
from modelscope.utils.constant import ConfigKeys, ModeKeys, ModelFile
from .ofa_trainer_utils import (AdjustLabelSmoothedCrossEntropyCriterion,
                                get_schedule)


@TRAINERS.register_module(module_name=Trainers.ofa_tasks)
class OFATrainer(EpochBasedTrainer):

    def __init__(self, model: str, cfg_file, work_dir, train_dataset,
                 eval_dataset, *args, **kwargs):
        model = Model.from_pretrained(model)
        model_dir = model.model_dir
        # cfg_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
        cfg = Config.from_file(cfg_file)
        # dataset = self._build_dataset_with_config(cfg)
        preprocessor = {
            ConfigKeys.train:
            OfaPreprocessor(
                model_dir=model_dir, mode=ModeKeys.TRAIN, no_collate=True),
            ConfigKeys.val:
            OfaPreprocessor(
                model_dir=model_dir, mode=ModeKeys.EVAL, no_collate=True),
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
        optimizer = build_optimizer(model, cfg=cfg.train.optimizer)
        scheduler_class, scheduler_args = get_schedule(cfg.train.lr_scheduler)
        if scheduler_class is not None:
            lr_scheduler = scheduler_class(**{'optimizer': optimizer},
                                           **scheduler_args)
        else:
            lr_scheduler = None
        collator = partial(
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
            cfg_file=cfg_file,
            model=model,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessor=preprocessor,
            optimizers=(optimizer, lr_scheduler),
            work_dir=work_dir,
            *args,
            **kwargs,
        )

    def train_step(self, model, inputs):
        model.train()
        model_outputs = model.forward(inputs)
        loss, sample_size, logging_output = self.criterion(
            model_outputs, inputs)
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

    def _build_dataset_with_config(self, cfg):
        if hasattr(cfg.dataset, 'hf_dataset'):
            dataset = load_dataset(
                cfg.dataset.script,
                data_files=cfg.dataset.hf_dataset,
                sep=cfg.dataset.sep,
            )
            dataset = MsDataset.from_hf_dataset(
                dataset.rename_columns(cfg.dataset.column_map))
            return dataset
        elif hasattr(cfg.dataset, 'ms_dataset'):
            dataset_d = dict()
            for key in cfg.dataset.ms_dataset.keys():
                dataset_d[key] = MsDataset.load(**cfg.dataset.ms_dataset[key])
                dataset_d[key] = MsDataset.from_hf_dataset(
                    dataset_d[key]._hf_ds.rename_columns(
                        cfg.dataset.column_map))
            return dataset_d
        else:
            raise NotImplementedError
