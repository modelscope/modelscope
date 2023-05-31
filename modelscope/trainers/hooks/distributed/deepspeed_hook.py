# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil

import deepspeed
import torch
from deepspeed import DeepSpeedEngine
from megatron_util import mpu, print_rank_0

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks import LoadCheckpointHook
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import (
    BestCkptSaverHook, CheckpointHook)
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.logger import get_logger
from ..checkpoint.checkpoint_processor import CheckpointProcessor
from ..lr_scheduler_hook import LrSchedulerHook, LrSchedulerProcessor
from ..optimizer.base import OptimizerHook, OptimizerProcessor


class DeepspeedProcessor(CheckpointProcessor, LrSchedulerProcessor,
                         OptimizerProcessor):

    _BIN_FILE_DIR = 'model'

    def rank_name(self):
        # TODO
        try:
            tp_world_size = mpu.get_tensor_model_parallel_world_size()
            if tp_world_size == 1:
                return ''
            mp_rank = mpu.get_tensor_model_parallel_rank()
            return '_mp_rank_{:02d}'.format(mp_rank)
        except (ImportError, AssertionError):
            return ''

    def get_bin_file(self):
        mp_rank = mpu.get_tensor_model_parallel_rank()
        rank = '{:02d}'.format(mp_rank)
        return f'mp_rank_{rank}_model_states.pt'

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None):
        model = trainer.unwrap_module(trainer.model)
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointProcessor.TRAINER_STATE_SUFFIX
        # Save pth file without model state_dict
        save_checkpoint(
            model, _train_state_file, None, None, meta=meta, with_model=False)

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        trainer.model.save_checkpoint(save_dir, prefix)

        bin_file = self.get_bin_file()
        src_file = os.path.join(checkpoint_path_prefix, bin_file)
        dest_file = os.path.join(output_dir, self._BIN_FILE_DIR, bin_file)
        if os.path.isfile(dest_file):
            os.unlink(dest_file)

        try:
            os.link(src_file, dest_file)
        except OSError as e:
            get_logger().error(
                f'Link {src_file} to {dest_file} error: {e}, '
                'changing to copy the bin file, this may case more space usage.'
            )
            shutil.copyfile(src_file, dest_file)

    def remove_checkpoints(self, trainer, checkpoint_path_prefix):
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointProcessor.TRAINER_STATE_SUFFIX
        if os.path.isfile(_train_state_file):
            os.remove(_train_state_file)

        shutil.rmtree(checkpoint_path_prefix, ignore_errors=True)

    def load_checkpoints(self, checkpoint_path_prefix, trainer, load_all_state,
                         strict):
        assert os.path.isdir(checkpoint_path_prefix)
        path = os.path.dirname(checkpoint_path_prefix)
        tag = os.path.basename(checkpoint_path_prefix)

        meta = {}
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointProcessor.TRAINER_STATE_SUFFIX
        if os.path.isfile(_train_state_file):
            meta = self.load_trainer_state(trainer, _train_state_file,
                                           load_all_state)

        if isinstance(trainer.model, DeepSpeedEngine):
            # DeepSpeedEngine is initialized
            trainer.model.load_checkpoint(
                path,
                tag,
                load_module_strict=strict,
                load_module_only=not load_all_state,
            )
        else:
            # in eval or prediction
            save_dir = checkpoint_path_prefix
            bin_file = self.get_bin_file()
            model_file = os.path.join(save_dir, bin_file)
            checkpoint = torch.load(
                model_file, map_location=lambda storage, loc: storage)
            checkpoint = checkpoint['module']
            model_dict = trainer.unwrap_module(trainer.model).state_dict()
            for key in checkpoint:
                if key not in model_dict.keys():
                    print_rank_0('Skip key: ' + key)
                else:
                    print_rank_0('Loading key: ' + key)
            trainer.unwrap_module(trainer.model).load_state_dict(
                checkpoint, strict=strict)
        return meta

    def backward(self, trainer, loss_keys, cumulative_iters, grad_clip):
        # assert cumulative_iters == 1, 'DeepSpeed only support cumulative_iters=1'
        # The `trainer.model` here is actually a deepspeed engine object.
        # backward step
        for k in loss_keys:
            loss = trainer.train_outputs[k]
            trainer.model.backward(loss)

        # update parameters
        trainer.model.step()

    def initialize_optimizer(self, trainer):
        pass

    def step(self, trainer):
        pass


@HOOKS.register_module(module_name=Hooks.DeepspeedHook)
class DeepspeedHook(Hook):
    PRIORITY = Priority.VERY_HIGH

    def __init__(self,
                 deepspeed_activation_checkpointing=True,
                 save_zero_checkpoint=False,
                 with_mpu=True):
        self.save_zero_checkpoint = save_zero_checkpoint
        self.deepspeed_activation_checkpointing = deepspeed_activation_checkpointing
        # TODO without mpu
        self.with_mpu = with_mpu
        assert with_mpu, 'DeepspeedHook now is only for mpu models.'

    def register_processor(self, trainer):
        processor = DeepspeedProcessor()
        optimizer_hook = trainer.get_hook(OptimizerHook)
        if len(optimizer_hook) > 0 and not isinstance(
                optimizer_hook[0].processor, DeepspeedProcessor):
            optimizer_hook[0].set_processor(processor)
        lr_schedular_hook = trainer.get_hook(LrSchedulerHook)
        if len(lr_schedular_hook) > 0 and not isinstance(
                lr_schedular_hook[0].processor, DeepspeedProcessor):
            lr_schedular_hook[0].set_processor(processor)
        ckpt_hook = trainer.get_hook(CheckpointHook)
        if len(ckpt_hook) > 0 and not isinstance(ckpt_hook[0].processor,
                                                 DeepspeedProcessor):
            ckpt_hook[0].set_processor(processor)
        best_ckpt_hook = trainer.get_hook(BestCkptSaverHook)
        if len(best_ckpt_hook) > 0 and not isinstance(
                best_ckpt_hook[0].processor, DeepspeedProcessor):
            best_ckpt_hook[0].set_processor(processor)
        load_ckpt_hook = trainer.get_hook(LoadCheckpointHook)
        if len(load_ckpt_hook) > 0 and not isinstance(
                load_ckpt_hook[0].processor, DeepspeedProcessor):
            load_ckpt_hook[0].set_processor(processor)

    def before_val(self, trainer):
        pass

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

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
            model = trainer.unwrap_module(trainer.model)
            deepspeed.checkpointing.configure(
                mpu,
                deepspeed_config=args.deepspeed_config,
                num_checkpoints=model.config.num_hidden_layers)

            mpu.checkpoint = deepspeed.checkpointing.checkpoint
            mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed
