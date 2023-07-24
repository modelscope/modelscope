# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import shutil

from modelscope.metainfo import Pipelines
from modelscope.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                         save_configuration)
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master


class CheckpointProcessor:

    TRAINER_STATE_SUFFIX = '_trainer_state.pth'

    MODEL_STATE_SUFFIX = '.pth'

    def prepare_output(self, trainer, output_dir):
        """Prepares the output of target folder.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer: The trainer instance.
            output_dir: The target folder used in inference.
        """
        config = trainer.cfg

        # override pipeline by tasks name after finetune done,
        # avoid case like fill mask pipeline with a text cls task
        if config['task'] in [
                getattr(Pipelines, attr) for attr in dir(Pipelines)
                if not attr.startswith('__')
        ]:
            # TODO a temp fix to avoid pipeline_name and task mismatch
            config['pipeline'] = {'type': config['task']}

        self.copy_files_and_dump_config(trainer, output_dir, config, '*.bin')

    @staticmethod
    def copy_files_and_dump_config(trainer, output_dir, config, bin_file):
        """Copy useful files to target output folder and dumps the target configuration.json.
        """
        model = trainer.unwrap_module(trainer.model)

        class SaveConfig:

            def __init__(self, output_dir, config):
                self.output_dir = output_dir
                self.config = config

            def __call__(self, _output_dir, _config):
                self.config = _config

            def save_config(self):
                save_configuration(self.output_dir, self.config)

        for pop_key in [
                'push_to_hub', 'hub_repo_id', 'hub_token', 'private_hub'
        ]:
            if config.safe_get('train.checkpoint.period.'
                               + pop_key) is not None:
                config.safe_get('train.checkpoint.period').pop(pop_key)
            if config.safe_get('train.checkpoint.best.' + pop_key) is not None:
                config.safe_get('train.checkpoint.best').pop(pop_key)

        save_config_fn = SaveConfig(output_dir, config)

        if hasattr(model, 'save_pretrained'):
            # Save pretrained of model, skip saving checkpoint
            model.save_pretrained(
                output_dir,
                bin_file,
                save_function=lambda *args, **kwargs: None,
                config=save_config_fn.config,
                save_config_function=save_config_fn)

        if trainer.train_preprocessor is not None:
            trainer.train_preprocessor.save_pretrained(
                output_dir,
                save_config_fn.config,
                save_config_function=save_config_fn)
        if trainer.eval_preprocessor is not None:
            trainer.eval_preprocessor.save_pretrained(
                output_dir,
                save_config_fn.config,
                save_config_function=save_config_fn)
        save_config_fn.save_config()

    @staticmethod
    def _bin_file(model):
        """Get bin file path.
        """
        default_bin_file = ModelFile.TORCH_MODEL_BIN_FILE
        if hasattr(model,
                   'model_dir') and ModelFile.TORCH_MODEL_FILE in os.listdir(
                       model.model_dir):
            default_bin_file = ModelFile.TORCH_MODEL_FILE
        return default_bin_file

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        """Save the state dict for trainer and model.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
            checkpoint_path_prefix(`str`): The saving dir with a prefix.
                like: /tmp/test/epoch_0
            output_dir(`str`): The output dir for inference.
            meta: (`dict`): The meta info needed to be saved into files.
            save_optimizers: (`bool`): Do save the optimizers state
        """
        model = trainer.unwrap_module(trainer.model)
        _model_file, _train_state_file = self._get_state_file_name(
            checkpoint_path_prefix)

        # Save pth file without model state_dict
        self.save_trainer_state(trainer, model, _train_state_file, meta,
                                save_optimizers)
        self.save_model_state(model, _model_file)
        self.link(model, _model_file, output_dir)

    def remove_checkpoints(self, trainer, checkpoint_path_prefix):
        """Remove obsolete checkpoint files.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
            checkpoint_path_prefix(`str`): The saving dir with a prefix.
                like: /tmp/test/epoch_0
        """
        _model_file, _train_state_file = self._get_state_file_name(
            checkpoint_path_prefix)
        if os.path.isfile(_train_state_file):
            os.remove(_train_state_file)

        if os.path.isfile(_model_file):
            os.remove(_model_file)

    def should_save_on_rank(self, trainer):
        """Used in ddp or other distributed training scenario, returns whether do saving in current rank.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
        """
        return is_master()

    def link(self, model, src_file, output_dir):
        """Links the src bin file to the output folder.

        Args:
            model: The model instance.
            src_file: The src bin file path.
            output_dir: The target folder used in inference.
        """

        bin_file = self._bin_file(model)
        dest_file = os.path.join(output_dir, bin_file)
        if os.path.isfile(dest_file):
            os.unlink(dest_file)

        try:
            os.link(src_file, dest_file)
        except OSError as e:
            get_logger().error(
                f'Link {src_file} to {dest_file} error: {e}, '
                'changing to copy the bin file, this may use more disk space.')
            shutil.copyfile(src_file, dest_file)

    def save_trainer_state(self, trainer, model, train_state_file, meta,
                           save_optimizers):
        """Save the trainer state, including optimizer/lr_scheduler's state dict, random states etc.

        Args:
            trainer: The trainer instance.
            model: The model instance.
            train_state_file: The target file name for saving trainer states.
            meta: Some extra meta info.
            save_optimizers: Save optimizers state or not.
        """
        save_checkpoint(
            model,
            train_state_file,
            trainer.optimizer if save_optimizers else None,
            trainer.lr_scheduler if save_optimizers else None,
            meta=meta,
            with_model=False)

    def save_model_state(self, model, model_file):
        """Save the model state.

        Args:
            model: The model instance.
            model_file: The target file name for saving model states.
        """
        save_checkpoint(
            model, model_file, None, None, meta=None, with_meta=False)

    def load_checkpoints(self, checkpoint_path_prefix, trainer, load_all_state,
                         strict):
        """Load checkpoint files of trainer state and model state.

        This is a strategic function which can be registered by other hook's function.

        Args:
            checkpoint_path_prefix(str): The checkpoint dir with prefix or a model state file.
                Example: '/tmp/test/epoch_0' or '/tmp/test/epoch_0.pth'
            trainer(`EpochBasedTrainer`): The trainer instance.
            load_all_state(`boolean`): Load all states (else load only module states).
            strict(`boolean`): If strict, any unmatched keys will cause an error.

        Returns:
            The meta info in json.
        """
        _model_file, _train_state_file = self._get_state_file_name(
            checkpoint_path_prefix)
        meta = {}
        if os.path.isfile(_train_state_file):
            meta = self.load_trainer_state(trainer, _train_state_file,
                                           load_all_state)
        else:
            print(f'No trainer state file {_train_state_file} found, skip.')
        self.load_model_state(trainer, _model_file, strict)
        return meta

    @staticmethod
    def load_trainer_state(trainer, train_state_file, load_all_state):
        """Load trainer state file.
        """

        optimizer = getattr(trainer, 'optimizer',
                            None) if load_all_state else None
        lr_scheduler = getattr(trainer, 'lr_scheduler',
                               None) if load_all_state else None
        return load_checkpoint(train_state_file, None, optimizer, lr_scheduler)

    def load_model_state(self, trainer, model_file, strict):
        """Load model state file.
        """
        return load_checkpoint(model_file,
                               trainer.unwrap_module(trainer.model), None,
                               None)

    @staticmethod
    def _get_state_file_name(checkpoint_path_prefix):
        """Get the default file name for state files.

        If the input is a checkpoint dir with prefix, this function will append suffix for both checkpoint files.
        If the input is an absolute file name, this function will return it as the model file name, and append
            suffix for the trainer file name.

        NOTE: a best checkpoint filename with float or int metric value inside
            will not be judged as having a extension file name. like: '/tmp/test/epoch_0_accuracy0.85'

        Args:
            checkpoint_path_prefix(`str`): The checkpoint dir with prefix or a model state file
            with extension file name. like: '/tmp/test/epoch_0'

        Returns:
              A tuple of model state file name and trainer state file name.
        """
        base, ext = os.path.splitext(checkpoint_path_prefix)
        if len(ext) == 0 or re.match(r'^\d+$', ext[1:]):
            return checkpoint_path_prefix + CheckpointProcessor.MODEL_STATE_SUFFIX, \
                   checkpoint_path_prefix + CheckpointProcessor.TRAINER_STATE_SUFFIX # noqa
        else:
            return checkpoint_path_prefix, base + CheckpointProcessor.TRAINER_STATE_SUFFIX.split(
                '.')[0] + '.' + ext[1:]
