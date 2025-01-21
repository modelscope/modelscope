# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import torch
from packaging import version

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks import Hook
from modelscope.trainers.hooks.builder import HOOKS
from .base import OptimizerHook, OptimizerProcessor


class ApexOptimizerProcessor(OptimizerProcessor):

    def __init__(self, opt_level):
        self.opt_level = opt_level

    def initialize_optimizer(self, trainer):
        from apex import amp

        if version.parse(torch.__version__) >= version.parse('1.9.0'):
            trainer.logger.warning(
                'ApexAMPOptimizerHook is only tested on torch version 1.8.x,'
                'if it works abnormally please consider downgrading your torch version to 1.8.x.'
            )

        logging.info('open fp16')
        # TODO: fix it should initialze amp with model not wrapper by DDP or DP
        model = trainer.unwrap_module(trainer.model)
        trainer.model, trainer.optimizer = amp.initialize(
            model, trainer.optimizer, opt_level=self.opt_level)

        trainer.optimizer.zero_grad()

    def backward(self, trainer, loss_keys, cumulative_iters, grad_clip):
        for k in loss_keys:
            trainer.train_outputs[k] /= cumulative_iters

        from apex import amp
        for k in loss_keys:
            with amp.scale_loss(trainer.train_outputs[k],
                                trainer.optimizer) as scaled_loss:
                scaled_loss.backward()

        if Hook.every_n_iters(trainer, cumulative_iters):
            if grad_clip is not None:
                OptimizerProcessor.clip_grads(trainer.model.parameters(),
                                              **grad_clip)

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()


@HOOKS.register_module(module_name=Hooks.ApexAMPOptimizerHook)
class ApexAMPOptimizerHook(Hook):
    """
    Fp16 optimizer, if torch version is less than 1.6.0,
    you must install apex (https://www.github.com/nvidia/apex) else use torch.cuda.amp by default

    Args:
        opt_level (str): "O0" and "O3" are not true mixed precision,
            but they are useful for establishing accuracy and speed baselines, respectively.
            "O1" and "O2" are different implementations of mixed precision.
            Try both, and see what gives the best speedup and accuracy for your model.
    """

    PRIORITY = OptimizerHook.PRIORITY

    def __init__(self, opt_level='O1', **kwargs):
        self.opt_level = opt_level

        try:
            from apex import amp
        except ImportError:
            raise ValueError(
                'apex not installed, please install apex from https://www.github.com/nvidia/apex.'
            )

    def register_processor(self, trainer):
        optimizer_hook = trainer.get_hook(OptimizerHook)
        if len(optimizer_hook) > 0 and type(
                optimizer_hook[0].processor) in (type(None),
                                                 OptimizerProcessor):
            optimizer_hook[0].set_processor(
                ApexOptimizerProcessor(self.opt_level))
