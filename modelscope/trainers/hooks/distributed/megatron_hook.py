import os
import shutil

import torch
from megatron_util import mpu

from modelscope.metainfo import Hooks
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import (
    BestCkptSaverHook, CheckpointHook, CheckpointProcessor)
from modelscope.trainers.hooks.checkpoint.load_checkpoint_hook import \
    LoadCheckpointHook
from modelscope.trainers.hooks.hook import Hook
from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint
from modelscope.utils.constant import DistributedParallelType
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from modelscope.utils.megatron_utils import is_megatron_initialized
from modelscope.utils.torch_utils import get_local_rank


class MpuProcessor(CheckpointProcessor):

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

    def get_bin_filename(self):
        mp_rank = mpu.get_tensor_model_parallel_rank()
        rank = '{:02d}'.format(mp_rank)
        return f'mp_rank_{rank}_model_states.pt'

    def should_save_on_rank(self, trainer):
        # TODO
        return (not torch.distributed.is_initialized()
                ) or mpu.get_data_parallel_rank() == 0

    def prepare_output(self, trainer, output_dir):
        config = trainer.cfg
        CheckpointProcessor.copy_files_and_dump_config(trainer, output_dir,
                                                       config,
                                                       self._BIN_FILE_DIR)
        os.makedirs(
            os.path.join(output_dir, self._BIN_FILE_DIR), exist_ok=True)

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
            model,
            _train_state_file,
            trainer.optimizer if save_optimizers else None,
            trainer.lr_scheduler if save_optimizers else None,
            meta=meta,
            with_model=False)

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        bin_file = self.get_bin_filename()
        prefix_bin_file = os.path.join(save_dir, prefix + '_' + bin_file)
        save_checkpoint(model, prefix_bin_file, with_meta=False)

        src_file = prefix_bin_file
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

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        bin_file = self.get_bin_filename()
        absolute_file = os.path.join(save_dir, prefix + '_' + bin_file)
        if os.path.isfile(absolute_file):
            os.remove(absolute_file)

    def load_checkpoints(self, checkpoint_path_prefix, trainer, load_all_state,
                         strict):
        model = trainer.unwrap_module(trainer.model)
        if os.path.isdir(checkpoint_path_prefix):
            save_dir = checkpoint_path_prefix
            bin_file = self.get_bin_filename()
            model_file = os.path.join(save_dir, bin_file)
            load_checkpoint(model_file, model, None, None)
        else:
            _train_state_file = checkpoint_path_prefix + self.rank_name(
            ) + CheckpointProcessor.TRAINER_STATE_SUFFIX
            meta = LoadCheckpointHook.load_trainer_state(
                trainer, _train_state_file, load_all_state)

            save_dir = os.path.dirname(checkpoint_path_prefix)
            prefix = os.path.basename(checkpoint_path_prefix)
            bin_file = self.get_bin_filename()

            model_file = os.path.join(save_dir, prefix + '_' + bin_file)
            load_checkpoint(model_file, model, None, None)
            return meta


@HOOKS.register_module(module_name=Hooks.MegatronHook)
class MegatronHook(Hook):

    _BIN_FILE_DIR = 'model'

    def __init__(self):
        self.wrapped = False

    def register_processor(self, trainer: EpochBasedTrainer):
        processor = MpuProcessor()
        ckpt_hook = trainer.get_hook(CheckpointHook)
        if len(ckpt_hook) > 0 and not isinstance(ckpt_hook[0].processor,
                                                 MpuProcessor):
            ckpt_hook[0].set_processor(processor)
        best_ckpt_hook = trainer.get_hook(BestCkptSaverHook)
        if len(best_ckpt_hook) > 0 and not isinstance(
                best_ckpt_hook[0].processor, MpuProcessor):
            best_ckpt_hook[0].set_processor(processor)
        load_ckpt_hook = trainer.get_hook(LoadCheckpointHook)
        if len(load_ckpt_hook) > 0 and not isinstance(
                load_ckpt_hook[0].processor, MpuProcessor):
            load_ckpt_hook[0].set_processor(processor)

    def after_init(self, trainer):
        assert is_megatron_initialized()
        local_rank = get_local_rank()
        trainer.device = create_device(f'cuda:{local_rank}')
        trainer.model.to(trainer.device)
        trainer.parallel_groups[
            DistributedParallelType.DP] = mpu.get_data_parallel_group()
        trainer.parallel_groups[DistributedParallelType.
                                TP] = mpu.get_tensor_model_parallel_group()
        trainer.parallel_groups[DistributedParallelType.
                                PP] = mpu.get_pipeline_model_parallel_group()

    def before_run(self, trainer):
        self.wrap_module(trainer)

    def before_val(self, trainer):
        self.wrap_module(trainer)

    def wrap_module(self, trainer):
        if trainer._dist:
            if not self.wrapped:
                trainer.model = trainer.to_parallel(trainer.model)
                self.wrapped = True
