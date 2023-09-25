import os
import shutil

from modelscope.metainfo import Hooks
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import (
    BestCkptSaverHook, CheckpointHook, CheckpointProcessor)
from modelscope.trainers.hooks.checkpoint.load_checkpoint_hook import \
    LoadCheckpointHook
from modelscope.trainers.hooks.hook import Hook
from modelscope.utils.checkpoint import save_configuration
from modelscope.utils.import_utils import is_swift_available


class SwiftCheckpointProcessor(CheckpointProcessor):

    _BIN_FILE_DIR = 'model'
    SWIFT_SAVE_SUFFIX = '_swift'

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
            if not is_swift_available():
                raise ValueError(
                    'Please install swift by `pip install ms-swift` to use SwiftHook.'
                )
            from swift import SwiftModel
            if isinstance(model, SwiftModel):
                _swift_output_dir = output_dir + SwiftCheckpointProcessor.SWIFT_SAVE_SUFFIX
                model.save_pretrained(
                    save_directory=_swift_output_dir,
                    safe_serialization=config.safe_get(
                        'train.checkpoint.safe_serialization', False),
                    adapter_name=config.safe_get(
                        'train.checkpoint.adapter_name', 'default'))
            else:
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

    def link_dir(self, source_dir, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)

    def save_swift_model_state(self, model, filename):
        model.save_pretrained(filename)

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        model = trainer.unwrap_module(trainer.model)
        _model_file, _train_state_file = self._get_state_file_name(
            checkpoint_path_prefix)
        _swift_save_dir = checkpoint_path_prefix + SwiftCheckpointProcessor.SWIFT_SAVE_SUFFIX
        _swift_output_dir = output_dir + SwiftCheckpointProcessor.SWIFT_SAVE_SUFFIX
        self.save_trainer_state(trainer, model, _train_state_file, meta,
                                save_optimizers)
        self.save_model_state(model, _model_file)
        self.link(model, _model_file, output_dir)
        self.save_swift_model_state(model, _swift_save_dir)
        self.link_dir(_swift_save_dir, _swift_output_dir)


@HOOKS.register_module(module_name=Hooks.SwiftHook)
class SwiftHook(Hook):

    _BIN_FILE_DIR = 'model'

    def __init__(self):
        pass

    def register_processor(self, trainer: EpochBasedTrainer):
        processor = SwiftCheckpointProcessor()
        ckpt_hook = trainer.get_hook(CheckpointHook)
        if len(ckpt_hook) > 0 and not isinstance(ckpt_hook[0].processor,
                                                 SwiftCheckpointProcessor):
            ckpt_hook[0].set_processor(processor)
        best_ckpt_hook = trainer.get_hook(BestCkptSaverHook)
        if len(best_ckpt_hook) > 0 and not isinstance(
                best_ckpt_hook[0].processor, SwiftCheckpointProcessor):
            best_ckpt_hook[0].set_processor(processor)
        load_ckpt_hook = trainer.get_hook(LoadCheckpointHook)
        if len(load_ckpt_hook) > 0 and not isinstance(
                load_ckpt_hook[0].processor, SwiftCheckpointProcessor):
            load_ckpt_hook[0].set_processor(processor)
