# Copyright 2020 The HuggingFace Team. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil

import math
import deepspeed
import torch
from functools import partialmethod
from deepspeed import DeepSpeedEngine
from megatron_util import mpu, print_rank_0

from modelscope.utils.torch_utils import get_local_rank, get_dist_info
from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.logger import get_logger
from .checkpoint_hook import CheckpointHook, LoadCheckpointHook
from modelscope.utils.constant import DistributedParallelType

# from accelerate.utils.deepspeed import HfDeepSpeedConfig
from transformers.deepspeed import HfTrainerDeepSpeedConfig

class DeepSpeedConfig(HfTrainerDeepSpeedConfig):
    """
    The `DeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = None
        self.mismatches = []

    def dtype(self):
        if self._dtype is None:
            raise ValueError("trainer_config_process() wasn't called yet to tell dtype")
        return self._dtype

    def is_auto(self, ds_key_long):
        val = self.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    fill_only = partialmethod(fill_match, must_match=False)

    def trainer_config_process(self, args):
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
        batch_size_per_gpu = args.train.dataloader.get("batch_size_per_gpu", 4)
        gradient_accumulation_steps = args.train.get("gradient_accumulation_steps", 8)
        workers_per_gpu = args.train.dataloader.get("workers_per_gpu", 2)
        clip_grad = args.train.get("clip_grad", 1.0)
        lr = args.train.optimizer.get("lr", 2e-5)
        adam_beta1 = args.train.optimizer.get("adam_beta1", 0.9)
        adam_beta2 = args.train.optimizer.get("adam_beta2", 0.999)
        adam_epsilon = args.train.optimizer.get("adam_epsilon", 1e-8)
        weight_decay = args.train.optimizer.get("weight_decay", 0.0)

        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = args.world_size * batch_size_per_gpu * gradient_accumulation_steps

        self.fill_match(
            "train_micro_batch_size_per_gpu", batch_size_per_gpu)
        self.fill_match("gradient_accumulation_steps", gradient_accumulation_steps)
        self.fill_match("train_batch_size", train_batch_size)
        self.fill_match("gradient_clipping", clip_grad)

        self.fill_match("optimizer.params.lr", lr)
        self.fill_match("optimizer.params.betas", [adam_beta1, adam_beta2])
        self.fill_match("optimizer.params.eps", adam_epsilon)
        self.fill_match("optimizer.params.weight_decay", weight_decay)

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_lr", lr)
        # total_num_steps - will get set in trainer_config_finalize

        args.fp16 = args.train.get("use_fp16", False)
        args.fp16_full_eval = args.train.get("use_fp16", False)
        args.fp16_backend = args.train.get("fp16_backend", "amp")
        # fp16
        if args.fp16 or args.fp16_full_eval:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None

        args.save_on_each_node = args.train.get("save_on_each_node", False)
        if args.save_on_each_node:
            # deepspeed uses shared storage by default. Let's override this setting if save_on_each_node == True
            self.config["checkpoint"] = self.config.get("checkpoint", {})
            self.config["checkpoint"]["use_node_local_storage"] = args.save_on_each_node

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        self.fill_match(
            "fp16.enabled",
            ((args.fp16 or args.fp16_full_eval) and fp16_backend == "amp"),
            "fp16|fp16_full_eval+fp16_backend(amp)",
        )

        args.fp16_opt_level = args.train.get("fp16_opt_level", None)
        args.fp16_opt_level = next((item["opt_level"] for item in args.train.hooks if item["type"] == "ApexAMPOptimizerHook"), args.fp16_opt_level)
        if not args.fp16_opt_level:
            args.fp16_opt_level = "O1"
        # apex: delegates amp work to apex (which needs to be available), but it cannot be used with any
        # ZeRO features
        self.fill_match("amp.enabled", fp16_backend == "apex", "fp16+fp16_backend(apex)")
        self.fill_match("amp.opt_level", args.fp16_opt_level, "fp16_opt_level")

        args.bf16 = args.train.get("bf16", False)
        self.fill_match("bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval")

        # deepspeed's default mode is fp16 unless there is a config that says differently
        if self.is_true("bf16.enabled"):
            self._dtype = torch.bfloat16
        elif self.is_false("fp16.enabled"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        # zero

        # deal with config keys that use `auto` value and rely on model's hidden_size
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]

        if len(hidden_size_auto_keys) > 0:
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, "hidden_sizes"):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model.config.hidden_sizes)
            else:
                raise ValueError(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    "`auto` values for these keys with an integer value of your choice."
                )

            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            if self.is_zero3():
                # automatically assign the optimal config values based on model config
                self.fill_only("zero_optimization.stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
                self.fill_only("zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)

        # scheduler
        warmup = args.train.optimizer["options"].get("warmup", {})
        warmup_steps = warmup.get("warmup_steps", 0)
        warmup_ratio = warmup.get("warmup_ratio", 0.0)
        warmup_steps = warmup_steps if warmup_steps > 0 else math.ceil(num_training_steps * warmup_ratio)
        self.fill_match("scheduler.params.total_num_steps", num_training_steps)
        self.fill_match("scheduler.params.warmup_num_steps", warmup_steps)


        if len(self.mismatches) > 0:
            mismatches = "\n".join(self.mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch TrainingArguments"
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )

def deepspeed_optim_sched(trainer, hf_deepspeed_config, num_training_steps):
    config = hf_deepspeed_config.config
    optimizer = None
    if "optimizer" not in config:
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the"
                " custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        optimizer = trainer.optimizer
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    if "scheduler" not in config:
        lr_scheduler = trainer.scheduler

    return optimizer, lr_scheduler


@HOOKS.register_module(module_name=Hooks.DeepspeedHook)
class DeepspeedHook(Hook):
    PRIORITY = Priority.VERY_HIGH
    _BIN_FILE_DIR = 'model'

    def __init__(self,
                 config,
                 deepspeed_activation_checkpointing=True,
                 save_zero_checkpoint=False,
                 with_mpu=True):
        self.save_zero_checkpoint = save_zero_checkpoint
        self.deepspeed_activation_checkpointing = deepspeed_activation_checkpointing
        # TODO without mpu
        self.with_mpu = with_mpu
        self.deepspeed_config = config
        #assert with_mpu, 'DeepspeedHook now is only for mpu models.'

    def register_strategy(self):
        Hook.overload(name='OptimizerHook.backward', function=self.backward)
        Hook.overload(
            name='OptimizerHook.initialize_optimizer', function=self.idle)
        Hook.overload(name='LrSchedulerHook.step', function=self.idle)
        Hook.overload(name='LrSchedulerHook.get_current_lr', function=self.get_current_lr)
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
        Hook.overload(
            name='DDPHook.wrap_module', function=self.wrap_module)
        if self.with_mpu:
            Hook.overload(
                name='CheckpointHook.should_save_on_rank',
                function=self.should_save_on_rank)

    def wrap_module(self, trainer):
        # deepspeed initializes its own ddp
        self.wrapped = True

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

    def get_current_lr(self, trainer):
        if isinstance(trainer.optimizer, torch.optim.Optimizer) or isinstance(trainer.optimizer, deepspeed.DeepSpeedOptimizer):
            lr = [group['lr'] for group in trainer.optimizer.param_groups]
        elif isinstance(trainer.optimizer, dict):
            lr = dict()
            for name, optim in trainer.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr


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

    def prepare_output(self, trainer, output_dir):
        config = trainer.cfg
        CheckpointHook.copy_files_and_dump_config(trainer, output_dir, config,
                                                  self._BIN_FILE_DIR)
        os.makedirs(
            os.path.join(output_dir, self._BIN_FILE_DIR), exist_ok=True)

    def before_val(self, trainer):
        pass

    def after_init(self, trainer):
        device_id = get_local_rank()
        trainer.device = f'cuda:{device_id}'
        #trainer.parallel_groups[DistributedParallelType.DP] = None

    def prepare_for_init(self, trainer):
        args = trainer.cfg
        _, args.world_size = get_dist_info()
        if os.path.exists(self.deepspeed_config):
            deepspeed_config = self.deepspeed_config
        else:
            deepspeed_config = os.path.join(trainer.model_dir,
                                            self.deepspeed_config)
        self.logger.info(f"Loading deepspeed config from {deepspeed_config}")

        gradient_accumulation_steps = args.train.get("gradient_accumulation_steps", 8)
        num_update_steps_per_epoch = trainer.iters_per_epoch // gradient_accumulation_steps
        max_steps = math.ceil(trainer._max_epochs * num_update_steps_per_epoch)

        ds_config = DeepSpeedConfig(deepspeed_config)
        ds_config.trainer_config_process(args)

        ds_config.trainer_config_finalize(args, trainer.model, max_steps)
        optimizer, lr_scheduler = deepspeed_optim_sched(trainer, ds_config, max_steps)
        config = ds_config.config
        return config, optimizer, lr_scheduler

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

        # deepspeed init

        config, optimizer, lr_scheduler = self.prepare_for_init(trainer)
        # TODO: 判断是否需要dist_init 和 mpu 而非写死;
        trainer.model, trainer.optimizer, _, trainer.lr_scheduler = deepspeed.initialize(
            model=trainer.model,
            optimizer=optimizer,
            config=config,
            lr_scheduler=lr_scheduler)
        trainer.model.save_zero_checkpoint = self.save_zero_checkpoint
