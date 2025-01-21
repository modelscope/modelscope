# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
import shutil
import tempfile
from functools import partial
from shutil import ignore_patterns
from typing import Callable, Dict, Optional, Tuple, Union

import json
import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import Dataset

from modelscope.hub.file_download import model_file_download
from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.parallel.utils import is_parallel
from modelscope.utils.config import Config
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ConfigKeys,
                                       Invoke, ModeKeys, ModelFile)
from .ofa_trainer_utils import (AdjustLabelSmoothedCrossEntropyCriterion,
                                get_schedule, recursive_overwrite)


@TRAINERS.register_module(module_name=Trainers.ofa)
class OFATrainer(EpochBasedTrainer):
    r"""
    OFA trainer for MaaS.

    Args:
        model (`str`): A model dir or a model id to be loaded
        cfg_file (`str`, **optional**, default to `None`):
            A config dir
        cfg_modify_fn (`Callable`, **optional**, default to `None`):
            A function which can rebuild the config file.
        arg_parse_fn (`Callable`, **optional**, default to `None`):
            Same as ``parse_fn`` in :obj:`Config.to_args`.
        data_collator (`Callable`, **optional**, default to `None`):
            The function to use to form a batch from a list of elements
            of `train_dataset` or `eval_dataset`.
        train_dataset (:obj:`MsDataset` or :obj:`Dataset`, **optional**, default to `None`):
            Dataset for training.
        eval_dataset (:obj:`MsDataset` or :obj:`Dataset`, **optional**, default to `None`):
            Dataset for evaluation.
        preprocessor (:obj:`Preprocessor`, **optional**, default to `None`):
            The optional preprocessor.
            NOTE: If the preprocessor has been called before the dataset fed into this trainer by user's custom code,
            this parameter should be None, meanwhile remove the 'preprocessor' key from the cfg_file.
            Else the preprocessor will be instantiated from the cfg_file or assigned from this parameter and
            this preprocessing action will be executed every time the dataset's __getitem__ is called.
        model_revision (`str`, **optional**, default to `None`):
            The revision used when the model_name_or_path is
                a model id of the remote hub. default `None`.
        seed (`int`, **optional**, default to `42`):
            The optional random seed for torch, cuda, numpy and random.
    """

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
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
        model = Model.from_pretrained(
            model, revision=model_revision, invoked_by=Invoke.TRAINER)
        model_dir = model.model_dir
        self.cfg_modify_fn = cfg_modify_fn

        work_dir = kwargs.get('work_dir', 'workspace')
        os.makedirs(work_dir, exist_ok=True)
        ignore_file_set = set()
        if cfg_file is not None:
            cfg_file = self.get_config_file(cfg_file)
            dst = os.path.abspath(
                os.path.join(work_dir, ModelFile.CONFIGURATION))
            src = os.path.abspath(cfg_file)
            if src != dst:
                shutil.copy(src, work_dir)
            ignore_file_set.add(ModelFile.CONFIGURATION)
        recursive_overwrite(
            model_dir, work_dir, ignore=ignore_patterns(*ignore_file_set))
        cfg_file = os.path.join(work_dir, ModelFile.CONFIGURATION)
        cfg = self.rebuild_config(Config.from_file(cfg_file))
        if cfg_modify_fn is not None:
            cfg = self.cfg_modify_fn(cfg)
            with open(cfg_file, 'w') as writer:
                json.dump(dict(cfg), fp=writer, indent=4)
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
            cfg_modify_fn=cfg_modify_fn,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessor=preprocessor,
            optimizers=optimizers,
            seed=seed,
            **kwargs,
        )

    def rebuild_config(self, cfg: Config):
        r"""
        rebuild config if `cfg_modify_fn` is not `None`.
        """
        if self.cfg_modify_fn is not None:
            cfg = self.cfg_modify_fn(cfg)
        return cfg

    def get_config_file(self, config_file: str):
        r"""
        support local file/ url or model_id with revision
        """
        if os.path.exists(config_file):
            return config_file
        else:
            temp_name = tempfile.TemporaryDirectory().name
            if len(config_file.split('#')) == 2:
                model_id = config_file.split('#')[0]
                revision = config_file.split('#')[-1].split('=')[-1]
            else:
                model_id = config_file
                revision = DEFAULT_MODEL_REVISION
            file_name = model_file_download(
                model_id,
                file_path=ModelFile.CONFIGURATION,
                revision=revision,
                cache_dir=temp_name)
            return file_name

    def train_step(self, model, inputs):
        r"""
        A single training step.

        step 1. Let the model in a trainable state.
        step 2. Execute the criterion function.
        step 3. Update the logging variable's value.
        step 4. Update the training result.

        Args:
            model (:obj:`torch.nn.Module` or :obj:`TorchModel`): The model to be run.
            inputs (`dict`): model inputs.
        """
        model = model.module if self._dist or is_parallel(model) else model
        model.train()
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
