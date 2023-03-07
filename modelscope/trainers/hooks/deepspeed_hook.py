# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil

import deepspeed
import torch
from deepspeed import DeepSpeedEngine
from megatron_util import mpu, print_rank_0

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.logger import get_logger
from .checkpoint_hook import CheckpointHook, LoadCheckpointHook
from .megatron_hook import MegatronHook


@HOOKS.register_module(module_name=Hooks.DeepspeedHook)
class DeepspeedHook(MegatronHook):
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

    def register_strategy(self):
        Hook.overload(name='OptimizerHook.backward', function=self.backward)
        Hook.overload(
            name='OptimizerHook.initialize_optimizer', function=self.idle)
        Hook.overload(name='LrSchedulerHook.step', function=self.idle)
        Hook.overload(
            name='CheckpointHook.save_checkpoints',
            function=self.save_checkpoints)
        Hook.overload(
            name='LoadCheckpointHook.load_checkpoints',
            function=self.load_checkpoints)
        Hook.overload(
            name='CheckpointHook.remove_checkpoints',
            function=self.remove_checkpoints)
        Hook.overload(
            name='CheckpointHook.prepare_output', function=self.prepare_output)
        if self.with_mpu:
            Hook.overload(
                name='CheckpointHook.should_save_on_rank',
                function=self.should_save_on_rank)

    def backward(self, trainer, loss_keys, cumulative_iters, grad_clip):
        # assert cumulative_iters == 1, 'DeepSpeed only support cumulative_iters=1'
        # The `trainer.model` here is actually a deepspeed engine object.
        # backward step
        for k in loss_keys:
            loss = trainer.train_outputs[k]
            trainer.model.backward(loss)

        # update parameters
        trainer.model.step()

    def idle(self, *args, **kwargs):
        pass

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_sub_dir,
                         meta=None):
        model = trainer.unwrap_module(trainer.model)
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointHook.TRAINER_STATE_SUFFIX
        # Save pth file without model state_dict
        save_checkpoint(
            model, _train_state_file, None, None, meta=meta, with_model=False)

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        trainer.model.save_checkpoint(save_dir, prefix)

        bin_file = self.get_bin_file()
        src_file = os.path.join(checkpoint_path_prefix, bin_file)
        dest_file = os.path.join(save_dir, output_sub_dir, self._BIN_FILE_DIR,
                                 bin_file)
        if os.path.isfile(dest_file):
            os.unlink(dest_file)

        os.link(src_file, dest_file)

    def remove_checkpoints(self, trainer, checkpoint_path_prefix):
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointHook.TRAINER_STATE_SUFFIX
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
        ) + CheckpointHook.TRAINER_STATE_SUFFIX
        if os.path.isfile(_train_state_file):
            meta = LoadCheckpointHook.load_trainer_state(
                trainer, _train_state_file, load_all_state)

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
