# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from types import MethodType

import deepspeed
from megatron_util import mpu

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks import (BestCkptSaverHook, CheckpointHook,
                                       LrSchedulerHook, NoneLrSchedulerHook,
                                       NoneOptimizerHook, OptimizerHook)
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.torch_utils import is_master
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module(module_name=Hooks.DeepspeedHook)
class DeepspeedHook(Hook):
    PRIORITY = Priority.VERY_HIGH

    def __init__(self,
                 deepspeed_activation_checkpointing=True,
                 save_zero_checkpoint=False,
                 loss_key='loss'):
        self.save_zero_checkpoint = save_zero_checkpoint
        self.loss_key = loss_key
        self.deepspeed_activation_checkpointing = deepspeed_activation_checkpointing

    def before_run(self, trainer):
        # deepspeed init
        args = trainer.cfg.train
        args.deepspeed_config = os.path.join(trainer.model_dir,
                                             args.deepspeed_config)

        trainer.model, _, _, _ = deepspeed.initialize(
            model=trainer.model,
            optimizer=trainer.optimizer,
            args=args,
            lr_scheduler=trainer.lr_scheduler,
            mpu=mpu,
            dist_init_required=False)
        trainer.model.save_zero_checkpoint = self.save_zero_checkpoint

        if self.deepspeed_activation_checkpointing:
            model = trainer.model
            while hasattr(model, 'module'):
                model = model.module
            deepspeed.checkpointing.configure(
                mpu,
                deepspeed_config=args.deepspeed_config,
                num_checkpoints=model.config.num_hidden_layers)

            mpu.checkpoint = deepspeed.checkpointing.checkpoint
            mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed

        # modify hooks
        for i, hook in enumerate(trainer._hooks):
            # backward & step
            if isinstance(hook, OptimizerHook):
                trainer._hooks[i] = NoneOptimizerHook()
            if isinstance(hook, LrSchedulerHook):
                trainer._hooks[i] = NoneLrSchedulerHook()

            # save checkpoint
            if isinstance(hook, CheckpointHook):

                def _save_checkpoint(self, trainer):
                    if self.by_epoch:
                        cur_save_dir = os.path.join(
                            self.save_dir,
                            f'{LogKeys.EPOCH}_{trainer.epoch + 1}')
                    else:
                        cur_save_dir = os.path.join(
                            self.save_dir,
                            f'{LogKeys.ITER}_{trainer.iter + 1}')
                    if (self.is_last_epoch(trainer)
                            and self.by_epoch) or (self.is_last_iter(trainer)
                                                   and not self.by_epoch):
                        cur_save_dir = os.path.join(self.save_dir,
                                                    ModelFile.TRAIN_OUTPUT_DIR)
                    trainer.model.save_checkpoint(cur_save_dir)

                trainer._hooks[i]._save_checkpoint = MethodType(
                    _save_checkpoint, trainer._hooks[i])

            if isinstance(hook, BestCkptSaverHook):

                def _save_checkpoint(self, trainer):
                    if self.by_epoch:
                        cur_save_dir = os.path.join(
                            self.save_dir,
                            f'best_{LogKeys.EPOCH}{trainer.epoch + 1}_{self.metric_key}{self._best_metric}'
                        )
                    else:
                        cur_save_dir = os.path.join(
                            self.save_dir,
                            f'best_{LogKeys.ITER}{trainer.iter + 1}_{self.metric_key}{self._best_metric}.pth'
                        )
                    trainer.model.save_checkpoint(cur_save_dir)
                    self._best_ckpt_file = cur_save_dir

                trainer._hooks[i]._save_checkpoint = MethodType(
                    _save_checkpoint, trainer._hooks[i])

    def after_train_iter(self, trainer):
        # The `trainer.model` here is actually a deepspeed engine object.
        # backward step
        loss = trainer.train_outputs[self.loss_key]
        trainer.model.backward(loss)

        # update parameters
        trainer.model.step()
