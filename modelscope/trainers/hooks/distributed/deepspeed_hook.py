# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2020 The HuggingFace Team. All rights reserved.
import math
import os
import shutil
from functools import partialmethod

import deepspeed
import torch
from deepspeed import DeepSpeedEngine
from megatron_util import mpu, print_rank_0
from transformers.deepspeed import HfTrainerDeepSpeedConfig

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks import LoadCheckpointHook
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import (
    BestCkptSaverHook, CheckpointHook)
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import \
    CheckpointProcessor
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.lr_scheduler_hook import (LrSchedulerHook,
                                                         LrSchedulerProcessor)
from modelscope.trainers.hooks.optimizer.base import (OptimizerHook,
                                                      OptimizerProcessor)
from modelscope.trainers.hooks.priority import Priority
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.constant import DistributedParallelType
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import (get_dist_info, get_local_rank,
                                          init_dist)


class DeepSpeedConfig(HfTrainerDeepSpeedConfig):
    """
    The `DeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def is_auto(self, ds_key_long):
        val = self.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == 'auto'

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage runs after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        # zero

        # deal with config keys that use `auto` value and rely on model's hidden_size
        hidden_size_based_keys = [
            'zero_optimization.reduce_bucket_size',
            'zero_optimization.stage3_prefetch_bucket_size',
            'zero_optimization.stage3_param_persistence_threshold',
        ]
        hidden_size_auto_keys = [
            x for x in hidden_size_based_keys if self.is_auto(x)
        ]

        if len(hidden_size_auto_keys) > 0:
            if hasattr(model.config, 'hidden_size'):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, 'hidden_sizes'):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model.config.hidden_sizes)
            else:
                raise ValueError(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f'in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing '
                    '`auto` values for these keys with an integer value of your choice.'
                )

            self.fill_only('zero_optimization.reduce_bucket_size',
                           hidden_size * hidden_size)
            if self.is_zero3():
                # automatically assign the optimal config values based on model config
                self.fill_only('zero_optimization.stage3_prefetch_bucket_size',
                               0.9 * hidden_size * hidden_size)
                self.fill_only(
                    'zero_optimization.stage3_param_persistence_threshold',
                    10 * hidden_size)

        # scheduler
        options = args.train.optimizer.get('options', {})
        warmup = options.get('warmup', {})
        warmup_steps = warmup.get('warmup_steps', 0)
        warmup_ratio = warmup.get('warmup_ratio', 0.0)
        warmup_steps = warmup_steps if warmup_steps > 0 else math.ceil(
            num_training_steps * warmup_ratio)
        self.fill_match('scheduler.params.total_num_steps', num_training_steps)
        self.fill_match('scheduler.params.warmup_num_steps', warmup_steps)

        if len(self.mismatches) > 0:
            mismatches = '\n'.join(self.mismatches)
            raise ValueError(
                'Please correct the following DeepSpeed config values that mismatch TrainingArguments'
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )


def deepspeed_optim_sched(trainer, hf_deepspeed_config, num_training_steps):
    config = hf_deepspeed_config.config
    optimizer = None
    if 'optimizer' not in config:
        if hf_deepspeed_config.is_offload():
            logger.info(
                'Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the'
                ' custom optimizer has both CPU and GPU implementation (except LAMB)'
            )

        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        optimizer = trainer.optimizer
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config['zero_allow_untested_optimizer'] = True

    lr_scheduler = None
    if 'scheduler' not in config:
        lr_scheduler = trainer.scheduler

    return optimizer, lr_scheduler


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

    def get_bin_filename(self, with_mpu=True):
        if not with_mpu:
            return 'pytorch_model.bin'
        else:
            mp_rank = mpu.get_tensor_model_parallel_rank()
            rank = '{:02d}'.format(mp_rank)
            return f'mp_rank_{rank}_model_states.pt'

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        model = trainer.unwrap_module(trainer.model)
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointProcessor.TRAINER_STATE_SUFFIX
        # Save pth file without model state_dict
        save_checkpoint(
            model, _train_state_file, None, None, meta=meta, with_model=False)

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        with_mpu = not mpu.is_unitialized()
        bin_file = self.get_bin_filename(with_mpu)
        src_file = os.path.join(checkpoint_path_prefix, bin_file)
        if self.zero_stage == 3 or with_mpu:
            trainer.model.save_checkpoint(save_dir, prefix)
        else:
            save_checkpoint(
                model, src_file, None, None, meta=None, with_meta=False)

        if self.zero_stage == 3:
            return
        if with_mpu:
            dest_file = os.path.join(output_dir, self._BIN_FILE_DIR, bin_file)
        else:
            dest_file = os.path.join(output_dir, bin_file)
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
            bin_file = self.get_bin_filename()
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
        # Optimizer step for deepspeed must be called on every step regardless of
        # the value of gradient accumulation iters
        trainer.model.step()

    def initialize_optimizer(self, trainer):
        pass

    def step(self, trainer):
        pass

    def should_save_on_rank(self, trainer):
        return True

    def get_current_lr(self, trainer):
        if isinstance(trainer.optimizer, torch.optim.Optimizer) or isinstance(
                trainer.optimizer, deepspeed.DeepSpeedOptimizer):
            lr = [group['lr'] for group in trainer.optimizer.param_groups]
        elif isinstance(trainer.optimizer, dict):
            lr = dict()
            for name, optim in trainer.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr


@HOOKS.register_module(module_name=Hooks.DeepspeedHook)
class DeepspeedHook(Hook):
    PRIORITY = Priority.VERY_HIGH

    def __init__(self,
                 config=None,
                 deepspeed_activation_checkpointing=True,
                 save_zero_checkpoint=False,
                 with_mpu=True,
                 zero_stage=None):
        self.save_zero_checkpoint = save_zero_checkpoint
        self.deepspeed_activation_checkpointing = deepspeed_activation_checkpointing
        self.with_mpu = with_mpu
        self.deepspeed_config = config
        if zero_stage is not None:
            assert zero_stage in (0, 1, 2,
                                  3), 'zero_stage must in (0, 1, 2, 3)!'
        self.zero_stage = zero_stage

    def register_processor(self, trainer):
        processor = DeepspeedProcessor()
        optimizer_hook = trainer.get_hook(OptimizerHook)
        if len(optimizer_hook) > 0 and not isinstance(
                optimizer_hook[0].processor, DeepspeedProcessor):
            optimizer_hook[0].set_processor(processor)
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

        lr_scheduler_hook = trainer.get_hook(LrSchedulerHook)
        if len(lr_scheduler_hook) > 0 and not isinstance(
                lr_scheduler_hook[0].processor, DeepspeedProcessor):
            lr_scheduler_hook[0].set_processor(processor)
        self.processor = processor

    def prepare_args(self, args):
        args.per_device_train_batch_size = args.train.dataloader.get(
            'batch_size_per_gpu', 4)
        args.max_grad_norm = args.train.get('clip_grad', 1.0)
        args.learning_rate = args.train.optimizer.get('lr', 2e-5)
        args.adam_beta1 = args.train.optimizer.get('adam_beta1', 0.9)
        args.adam_beta2 = args.train.optimizer.get('adam_beta2', 0.999)
        args.adam_epsilon = args.train.optimizer.get('adam_epsilon', 1e-8)
        args.weight_decay = args.train.optimizer.get('weight_decay', 0.0)
        args.fp16 = args.train.get('use_fp16', False)
        args.fp16_full_eval = args.train.get('use_fp16', False)
        args.fp16_backend = args.train.get('fp16_backend', 'amp')
        args.save_on_each_node = args.train.get('save_on_each_node', False)
        args.fp16_opt_level = args.train.get('fp16_opt_level', None)
        args.fp16_opt_level = next((item.get('opt_level', args.fp16_opt_level)
                                    for item in args.train.hooks
                                    if item['type'] == 'ApexAMPOptimizerHook'),
                                   args.fp16_opt_level)
        if not args.fp16_opt_level:
            args.fp16_opt_level = 'O1'
        args.bf16 = args.train.get('bf16', False)

    def get_deepspeed_config(self, trainer, args, max_steps):
        _, args.world_size = get_dist_info()
        self.prepare_args(args)
        if os.path.exists(self.deepspeed_config):
            deepspeed_config = self.deepspeed_config
        else:
            deepspeed_config = os.path.join(trainer.model_dir,
                                            self.deepspeed_config)
        if not os.path.exists(deepspeed_config):
            raise RuntimeError(
                f'No such DeepSpeed json config file: {self.deepspeed_config}.'
            )
        self.logger.info(f'Loading deepspeed config from {deepspeed_config}')

        ds_config = DeepSpeedConfig(deepspeed_config)
        ds_config.trainer_config_process(args)

        ds_config.trainer_config_finalize(args, trainer.model, max_steps)
        return ds_config

    def after_init(self, trainer):
        init_dist('pytorch')
        local_rank = get_local_rank()
        trainer.device = create_device(f'cuda:{local_rank}')
        trainer.model.to(trainer.device)
        trainer.parallel_groups[DistributedParallelType.DP] = None

    def before_val(self, trainer):
        pass

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

        # deepspeed init
        args = trainer.cfg
        args.gradient_accumulation_steps = args.train.optimizer.get(
            'options', {}).get('cumulative_iters', 1)
        num_update_steps_per_epoch = trainer.iters_per_epoch // args.gradient_accumulation_steps
        max_steps = math.ceil(trainer._max_epochs * num_update_steps_per_epoch)

        ds_config = self.get_deepspeed_config(trainer, args, max_steps)

        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, ds_config, max_steps)
        config = ds_config.config
        if self.zero_stage is not None:
            config['zero_optimization']['stage'] = self.zero_stage
        self.processor.zero_stage = config['zero_optimization'].get('stage', 0)

        trainer.model, trainer.optimizer, _, trainer.lr_scheduler = deepspeed.initialize(
            model=trainer.model,
            optimizer=optimizer,
            config=config,
            lr_scheduler=lr_scheduler)
