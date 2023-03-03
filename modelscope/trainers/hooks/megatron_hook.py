import os

import torch
from megatron_util import mpu

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint
from .checkpoint_hook import CheckpointHook, LoadCheckpointHook


@HOOKS.register_module(module_name=Hooks.MegatronHook)
class MegatronHook(Hook):

    _BIN_FILE_DIR = 'model'

    def register_strategy(self):
        Hook.overload(
            name='CheckpointHook.should_save_on_rank',
            function=self.should_save_on_rank)
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

    def should_save_on_rank(self, trainer):
        # TODO
        return (not torch.distributed.is_initialized()
                ) or mpu.get_data_parallel_rank() == 0

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
                         output_sub_dir,
                         meta=None):
        model = trainer.unwrap_module(trainer.model)
        _train_state_file = checkpoint_path_prefix + self.rank_name(
        ) + CheckpointHook.TRAINER_STATE_SUFFIX
        # Save pth file without model state_dict
        save_checkpoint(
            model,
            _train_state_file,
            trainer.optimizer,
            trainer.lr_scheduler,
            meta=meta,
            with_model=False)

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        bin_file = self.get_bin_file()
        prefix_bin_file = os.path.join(save_dir, prefix + '_' + bin_file)
        save_checkpoint(model, prefix_bin_file, with_meta=False)

        src_file = prefix_bin_file
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

        save_dir = os.path.dirname(checkpoint_path_prefix)
        prefix = os.path.basename(checkpoint_path_prefix)
        bin_file = self.get_bin_file()
        absolute_file = os.path.join(save_dir, prefix + '_' + bin_file)
        if os.path.isfile(absolute_file):
            os.remove(absolute_file)

    def load_checkpoints(self, checkpoint_path_prefix, trainer, load_all_state,
                         strict):
        model = trainer.unwrap_module(trainer.model)
        if os.path.isdir(checkpoint_path_prefix):
            save_dir = checkpoint_path_prefix
            bin_file = self.get_bin_file()
            model_file = os.path.join(save_dir, bin_file)
            load_checkpoint(model_file, model, None, None)
        else:
            _train_state_file = checkpoint_path_prefix + self.rank_name(
            ) + CheckpointHook.TRAINER_STATE_SUFFIX
            meta = LoadCheckpointHook.load_trainer_state(
                trainer, _train_state_file, load_all_state)

            save_dir = os.path.dirname(checkpoint_path_prefix)
            prefix = os.path.basename(checkpoint_path_prefix)
            bin_file = self.get_bin_file()

            model_file = os.path.join(save_dir, prefix + '_' + bin_file)
            load_checkpoint(model_file, model, None, None)
            return meta

    def prepare_output(self, trainer, output_dir):
        config = trainer.cfg.to_dict()
        CheckpointHook.copy_files_and_dump_config(trainer, output_dir, config,
                                                  self._BIN_FILE_DIR)
        os.makedirs(
            os.path.join(output_dir, self._BIN_FILE_DIR), exist_ok=True)
