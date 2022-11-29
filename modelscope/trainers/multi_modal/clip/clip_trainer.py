# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import Dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.models.multi_modal.clip.model import convert_models_to_fp32
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.multi_modal import CLIPPreprocessor
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ConfigKeys,
                                       Invoke, ModeKeys)
from .clip_trainer_utils import get_loss, get_optimizer_params, get_schedule


def exclude(n):
    return 'bn' in n or 'ln' in n or 'bias' in n or 'logit_scale' in n


def include(n):
    return not exclude(n)


@TRAINERS.register_module(module_name=Trainers.clip_multi_modal_embedding)
class CLIPTrainer(EpochBasedTrainer):

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
        model = Model.from_pretrained(
            model, revision=model_revision, invoked_by=Invoke.TRAINER)
        # for training & eval, we convert the model from FP16 back to FP32
        # to compatible with modelscope amp training
        convert_models_to_fp32(model)
        cfg = Config.from_file(cfg_file)
        if 'work_dir' not in kwargs or len(kwargs['work_dir']) == 0:
            work_dir = cfg.train.work_dir
        else:
            work_dir = kwargs['work_dir']

        # fetch the model name of CLIP model (base, large or large-336)
        model_name = cfg.pretrained_model.model_name

        # world size
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        # train step, optimizer and lr_scheduler
        epoch_steps = math.ceil(
            len(train_dataset) /  # noqa
            (cfg.train.dataloader.batch_size_per_gpu * world_size))  # noqa
        cfg.train.lr_scheduler.num_train_steps = epoch_steps * cfg.train.max_epochs

        if optimizers[0] is None:
            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [
                p for n, p in named_parameters
                if exclude(n) and p.requires_grad
            ]
            rest_params = [
                p for n, p in named_parameters
                if include(n) and p.requires_grad
            ]
            optimizer_hparams = get_optimizer_params(
                model_name, cfg)  # lr, wd, beta1, beta2, eps
            optimizer_args = {
                'params': [
                    {
                        'params': gain_or_bias_params,
                        'weight_decay': 0.
                    },
                    {
                        'params': rest_params,
                        'weight_decay': optimizer_hparams['weight_decay']
                    },
                ],
                'lr':
                optimizer_hparams['lr'],
                'betas':
                (optimizer_hparams['beta1'], optimizer_hparams['beta2']),
                'eps':
                optimizer_hparams['eps'],
            }
            optimizer = build_optimizer(
                model, cfg=cfg.train.optimizer, default_args=optimizer_args)
        else:
            optimizer = optimizers[0]

        if optimizers[1] is None:
            lr_scheduler = get_schedule(optimizer, cfg.train.lr_scheduler)
        else:
            lr_scheduler = optimizers[1]
        optimizers = (optimizer, lr_scheduler)

        # loss module
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        self.loss_img = loss_img.cuda(int(os.environ.get('LOCAL_RANK', 0)))
        self.loss_txt = loss_txt.cuda(int(os.environ.get('LOCAL_RANK', 0)))
        self.loss_cfg = cfg.train.loss_cfg

        # launcher and use_fp16
        if 'launcher' not in kwargs and cfg.train.get('launcher', None):
            kwargs['launcher'] = cfg.train.launcher
        if 'use_fp16' not in kwargs and cfg.train.get('use_fp16', False):
            kwargs['use_fp16'] = cfg.train.use_fp16

        # preprocessor
        if preprocessor is None:
            preprocessor = {
                ConfigKeys.train:
                CLIPPreprocessor(
                    model_dir=work_dir,
                    mode=ModeKeys.TRAIN,
                    tokenizer=model.tokenizer,
                    resolution=model.model_info['image_resolution']),
                ConfigKeys.val:
                CLIPPreprocessor(
                    model_dir=work_dir,
                    mode=ModeKeys.EVAL,
                    tokenizer=model.tokenizer,
                    resolution=model.model_info['image_resolution']),
            }

        # dataset related
        self.dataset_cfg = cfg.dataset
        if hasattr(self.dataset_cfg, 'column_map'):
            # cases where dataset key names are not "img" and "text"
            img_key_name = getattr(self.dataset_cfg.column_map, 'img', 'img')
            preprocessor[ConfigKeys.train].set_input_img_key(img_key_name)
            preprocessor[ConfigKeys.val].set_input_img_key(img_key_name)
            text_key_name = getattr(self.dataset_cfg.column_map, 'text',
                                    'text')
            preprocessor[ConfigKeys.train].set_input_text_key(text_key_name)
            preprocessor[ConfigKeys.val].set_input_text_key(text_key_name)
        self.global_batch_size = cfg.train.dataloader.batch_size_per_gpu * world_size

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
        inputs['mode'] = ModeKeys.TRAIN
        model_outputs = model.forward(
            inputs
        )  # {OutputKeys.IMG_EMBEDDING: Tensor(batch_size, dim), OutputKeys.TEXT_EMBEDDING: Tensor(batch_size, dim)}
        loss = get_loss(model_outputs, self.loss_img, self.loss_txt,
                        self.loss_cfg)
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
            unwrapped_model = getattr(model, 'module', model)
            log_vars[
                'logit_scale'] = unwrapped_model.clip_model.logit_scale.data.clone(
                ).item()  # noqa
            log_vars['global_batch_size'] = int(self.global_batch_size)
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])
        self.train_outputs = train_outputs
