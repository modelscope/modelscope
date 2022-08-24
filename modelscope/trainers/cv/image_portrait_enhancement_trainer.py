# Copyright (c) Alibaba, Inc. and its affiliates.
from collections.abc import Mapping

import torch
from torch import distributed as dist

from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger


@TRAINERS.register_module(module_name=Trainers.image_portrait_enhancement)
class ImagePortraitEnhancementTrainer(EpochBasedTrainer):

    def train_step(self, model, inputs):
        """ Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`TorchModel`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # EvaluationHook will do evaluate and change mode to val, return to train mode
        # TODO: find more pretty way to change mode
        self.d_reg_every = self.cfg.train.get('d_reg_every', 16)
        self.g_reg_every = self.cfg.train.get('g_reg_every', 4)
        self.path_regularize = self.cfg.train.get('path_regularize', 2)
        self.r1 = self.cfg.train.get('r1', 10)

        train_outputs = dict()
        self._mode = ModeKeys.TRAIN
        # call model forward but not __call__ to skip postprocess
        if isinstance(inputs, Mapping):
            d_loss = model._train_forward_d(**inputs)
        else:
            d_loss = model._train_forward_d(inputs)
        train_outputs['d_loss'] = d_loss

        model.discriminator.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()

        if self._iter % self.d_reg_every == 0:

            if isinstance(inputs, Mapping):
                r1_loss = model._train_forward_d_r1(**inputs)
            else:
                r1_loss = model._train_forward_d_r1(inputs)
            train_outputs['r1_loss'] = r1_loss

            model.discriminator.zero_grad()
            (self.r1 / 2 * r1_loss * self.d_reg_every).backward()

            self.optimizer_d.step()

        if isinstance(inputs, Mapping):
            g_loss = model._train_forward_g(**inputs)
        else:
            g_loss = model._train_forward_g(inputs)
        train_outputs['g_loss'] = g_loss

        model.generator.zero_grad()
        g_loss.backward()
        self.optimizer.step()

        path_loss = 0
        if self._iter % self.g_reg_every == 0:
            if isinstance(inputs, Mapping):
                path_loss = model._train_forward_g_path(**inputs)
            else:
                path_loss = model._train_forward_g_path(inputs)
            train_outputs['path_loss'] = path_loss

            model.generator.zero_grad()
            weighted_path_loss = self.path_regularize * self.g_reg_every * path_loss

            weighted_path_loss.backward()

            self.optimizer.step()

        model.accumulate()

        if not isinstance(train_outputs, dict):
            raise TypeError('"model.forward()" must return a dict')

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

    def create_optimizer_and_scheduler(self):
        """ Create optimizer and lr scheduler

        We provide a default implementation, if you want to customize your own optimizer
        and lr scheduler, you can either pass a tuple through trainer init function or
        subclass this class and override this method.


        """
        optimizer, lr_scheduler = self.optimizers
        if optimizer is None:
            optimizer_cfg = self.cfg.train.get('optimizer', None)
        else:
            optimizer_cfg = None
        optimizer_d_cfg = self.cfg.train.get('optimizer_d', None)

        optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
            optimizer = build_optimizer(
                self.model.generator, cfg=optimizer_cfg)
        if optimizer_d_cfg is not None:
            optimizer_d = build_optimizer(
                self.model.discriminator, cfg=optimizer_d_cfg)

        lr_options = {}
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.optimizer_d = optimizer_d
        return self.optimizer, self.lr_scheduler, optim_options, lr_options
