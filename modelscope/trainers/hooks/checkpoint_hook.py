# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import re
from shutil import rmtree

import numpy as np
import torch
from packaging import version

from modelscope import __version__
from modelscope.metainfo import Hooks, Pipelines
from modelscope.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                         save_configuration)
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.megatron_utils import is_megatron_initialized
from modelscope.utils.torch_utils import is_master
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module(module_name=Hooks.CheckpointHook)
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The frequency to save model. If `by_epoch=True`,
            it means the number of epochs, else means the number of iterations
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
        save_optimizer (bool): Whether to save optimizer state dict.  Default: True.
        save_dir (str): The directory to save checkpoints. If is None, use `trainer.work_dir`
        output_sub_dir (str): The sub folder under the `save_dir` to save the output checkpoint for inference.
            Default 'output'.
        save_last (bool): Whether to save the last checkpoint. Default: True.
        max_checkpoint_num (int): The max number of checkpoint files, default None which means never delete anything.
            If the number exceeding the limit, earlier checkpoints will be deleted first.
    """

    PRIORITY = Priority.LOW

    TRAINER_STATE_SUFFIX = '_trainer_state.pth'

    MODEL_STATE_SUFFIX = '.pth'

    def __init__(self,
                 interval=0,
                 by_epoch=True,
                 save_optimizer=True,
                 save_dir=None,
                 output_sub_dir=ModelFile.TRAIN_OUTPUT_DIR,
                 save_last=True,
                 max_checkpoint_num=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_dir = save_dir
        self.output_sub_dir = output_sub_dir
        self.save_last = save_last
        self.rng_state = None
        self.max_checkpoint_num = None
        if max_checkpoint_num is not None:
            self.max_checkpoint_num = max(int(max_checkpoint_num), 1)
        self.history_checkpoints = []

    def before_run(self, trainer):
        if not self.save_dir:
            self.save_dir = trainer.work_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

        if is_master():
            output_dir = os.path.join(self.save_dir, self.output_sub_dir)
            # only global master prepares the output folder
            self.prepare_output(trainer, output_dir)
            self.logger.info(f'Checkpoints will be saved to {self.save_dir}')

    def after_train_epoch(self, trainer):
        if not self.by_epoch:
            return

        if self._should_save(trainer) and self.should_save_on_rank(trainer):
            if is_master():
                self.logger.info(
                    f'Saving checkpoint at {trainer.epoch + 1} epoch')
            self._save_checkpoint(trainer)

    def after_train_iter(self, trainer):
        if self.by_epoch:
            return

        if self._should_save(trainer) and self.should_save_on_rank(trainer):
            if is_master():
                self.logger.info(
                    f'Saving checkpoint at {trainer.iter + 1} epoch')
            self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer):
        """Save checkpoint files and remove obsolete ones
        """

        if self.by_epoch:
            checkpoint_path_prefix = os.path.join(
                self.save_dir, f'{LogKeys.EPOCH}_{trainer.epoch + 1}')
        else:
            checkpoint_path_prefix = os.path.join(
                self.save_dir, f'{LogKeys.ITER}_{trainer.iter + 1}')

        meta = self._create_training_state(trainer)
        self.save_checkpoints(trainer, checkpoint_path_prefix,
                              self.output_sub_dir, meta)
        self.history_checkpoints.append(checkpoint_path_prefix)
        self._remove_obsolete_checkpoints(trainer)

    def _remove_obsolete_checkpoints(self, trainer):
        if self.max_checkpoint_num is not None and \
                len(self.history_checkpoints) > self.max_checkpoint_num:
            history_checkpoints = [ckpt for ckpt in self.history_checkpoints]
            self.history_checkpoints.clear()
            for i, checkpoint_path_prefix in enumerate(history_checkpoints):
                if i < len(history_checkpoints) - self.max_checkpoint_num:
                    self.logger.info(
                        f'deleting checkpoint: {checkpoint_path_prefix}')
                    self.remove_checkpoints(
                        trainer, checkpoint_path_prefix=checkpoint_path_prefix)
                else:
                    self.history_checkpoints.append(checkpoint_path_prefix)

    def _should_save(self, trainer):
        if self.by_epoch:
            check_last = self.is_last_epoch
            check_frequency = self.every_n_epochs
        else:
            check_last = self.is_last_iter
            check_frequency = self.every_n_iters

        if check_frequency(trainer,
                           self.interval) or (self.save_last
                                              and check_last(trainer)):
            return True
        return False

    def _create_training_state(self, trainer):
        self.rng_state = {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'cpu': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
        }

        # keep epoch/iter/inner_iter/random_state
        meta = {
            'epoch': trainer.epoch,
            'iter': trainer.iter + 1,
            'inner_iter': trainer.inner_iter + 1,
            'rng_state': self.rng_state,
        }

        # keep hooks state
        i = 0
        for hook in trainer.hooks:
            if hasattr(hook, 'state_dict') and getattr(hook, '_should_save',
                                                       True):
                meta[f'{hook.__class__}-{i}'] = hook.state_dict()
                i += 1

        return meta

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

    @Hook.overload_func(name='CheckpointHook.prepare_output')
    def prepare_output(self, trainer, output_dir):
        """Prepares the output of target folder.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer: The trainer instance.
            output_dir: The target folder used in inference.
        """
        model = trainer.unwrap_module(trainer.model)
        config = trainer.cfg.to_dict()

        # override pipeline by tasks name after finetune done,
        # avoid case like fill mask pipeline with a text cls task
        if config['task'] in [
                getattr(Pipelines, attr) for attr in dir(Pipelines)
                if not attr.startswith('__')
        ]:
            # TODO a temp fix to avoid pipeline_name and task mismatch
            config['pipeline'] = {'type': config['task']}

        self.copy_files_and_dump_config(trainer, output_dir, config,
                                        self._bin_file(model))

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

        os.link(src_file, dest_file)

    def save_trainer_state(self, trainer, model, train_state_file, meta):
        """Save the trainer state, including optimizer/lr_scheduler's state dict, random states etc.

        Args:
            trainer: The trainer instance.
            model: The model instance.
            train_state_file: The target file name for saving trainer states.
            meta: Some extra meta info.
        """
        save_checkpoint(
            model,
            train_state_file,
            trainer.optimizer,
            trainer.lr_scheduler,
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

    @Hook.overload_func(name='CheckpointHook.save_checkpoints')
    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_sub_dir,
                         meta=None):
        """Save the state dict for trainer and model.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
            checkpoint_path_prefix(`str`): The saving dir with a prefix.
                like: /tmp/test/epoch_0
            output_sub_dir(`str`): The sub-dir in the saving dir used in inference.
            meta: (`dict`): The meta info needed to be saved into files.
        """
        model = trainer.unwrap_module(trainer.model)
        _model_file, _train_state_file = _get_state_file_name(
            checkpoint_path_prefix)

        # Save pth file without model state_dict
        self.save_trainer_state(trainer, model, _train_state_file, meta)
        self.save_model_state(model, _model_file)
        output_dir = os.path.join(self.save_dir, output_sub_dir)
        self.link(model, _model_file, output_dir)

    @Hook.overload_func(name='CheckpointHook.remove_checkpoints')
    def remove_checkpoints(self, trainer, checkpoint_path_prefix):
        """Remove obsolete checkpoint files.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
            checkpoint_path_prefix(`str`): The saving dir with a prefix.
                like: /tmp/test/epoch_0
        """
        _model_file, _train_state_file = _get_state_file_name(
            checkpoint_path_prefix)
        if os.path.isfile(_train_state_file):
            os.remove(_train_state_file)

        if os.path.isfile(_model_file):
            os.remove(_model_file)

    @Hook.overload_func(name='CheckpointHook.should_save_on_rank')
    def should_save_on_rank(self, trainer):
        """Used in ddp or other distributed training scenario, returns whether do saving in current rank.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
        """
        return is_master()


@HOOKS.register_module(module_name=Hooks.BestCkptSaverHook)
class BestCkptSaverHook(CheckpointHook):
    """
    Save best checkpoints hook.

    Args:
        metric_key (str): Metric key to compare rule for best score.
        rule (str): Comparison rule for best score. Support "max" and "min". If rule is "max", the checkpoint
            at the maximum `metric_key` will be saved, If rule is "min", the checkpoint at the minimum `metric_key`
            will be saved.
        by_epoch (bool): Save best checkpoints by epoch or by iteration.
        save_optimizer (bool): Whether to save optimizer state dict.  Default: True.
        save_dir (str): Output directory to save best checkpoint.
        output_sub_dir (str): The sub folder under the `save_dir` to save the output checkpoint for inference.
            Default 'output_best'.
        restore_best (bool): Whether to restore the best checkpoint after training.
        max_checkpoint_num (int): The max number of checkpoint files, default None which means never delete anything.
            If the number exceeding the limit, checkpoints with worse metric will be deleted, which is judged by the
            `rule` and `metric_key` arguments.
    """

    PRIORITY = Priority.LOW
    rule_map = {'max': lambda x, y: x > y, 'min': lambda x, y: x < y}

    def __init__(self,
                 metric_key,
                 rule='max',
                 by_epoch=True,
                 save_optimizer=True,
                 save_dir=None,
                 output_sub_dir=ModelFile.TRAIN_BEST_OUTPUT_DIR,
                 save_file_name=None,
                 restore_best=False,
                 max_checkpoint_num=1,
                 interval=0,
                 **kwargs):
        assert rule in ['max', 'min'], 'Only support "max" or "min" rule now.'
        super().__init__(
            interval=interval,
            by_epoch=by_epoch,
            save_optimizer=save_optimizer,
            save_dir=save_dir,
            output_sub_dir=output_sub_dir,
            max_checkpoint_num=max_checkpoint_num,
            **kwargs,
        )
        self.metric_key = metric_key
        self.rule = rule
        self._best_metric = None
        self._best_ckpt_file = None
        self.save_file_name = save_file_name
        self.restore_best = restore_best
        self.history_checkpoints = set()

    def _should_save(self, trainer):
        return self._is_best_metric(trainer.metric_values)

    def _is_best_metric(self, metric_values):
        if metric_values is None:
            return False

        if self.metric_key not in metric_values:
            raise ValueError(
                f'Not find metric_key: {self.metric_key} in {metric_values}')

        if self._best_metric is None:
            self._best_metric = metric_values[self.metric_key]
            return True
        else:
            compare_fn = self.rule_map[self.rule]
            if compare_fn(metric_values[self.metric_key], self._best_metric):
                self._best_metric = metric_values[self.metric_key]
                return True
        return False

    def _save_checkpoint(self, trainer):
        checkpoint_path_prefix = self.save_file_name
        if checkpoint_path_prefix is None:
            if self.by_epoch:
                checkpoint_path_prefix = os.path.join(
                    self.save_dir,
                    f'best_{LogKeys.EPOCH}{trainer.epoch + 1}_{self.metric_key}{self._best_metric}'
                )
            else:
                checkpoint_path_prefix = os.path.join(
                    self.save_dir,
                    f'best_{LogKeys.ITER}{trainer.iter + 1}_{self.metric_key}{self._best_metric}'
                )
        else:
            checkpoint_path_prefix = os.path.join(self.save_dir,
                                                  checkpoint_path_prefix)

        self._best_ckpt_file = checkpoint_path_prefix
        meta = self._create_training_state(trainer)
        self.save_checkpoints(trainer, checkpoint_path_prefix,
                              self.output_sub_dir, meta)
        self.history_checkpoints.add(checkpoint_path_prefix)
        self._remove_obsolete_checkpoints(trainer)

    def _remove_obsolete_checkpoints(self, trainer):

        def extract_metric_from_filename(name1):
            metric1 = float(name1.split(self.metric_key)[1])
            if self.rule == 'max':
                return -metric1
            else:
                return metric1

        if self.max_checkpoint_num is not None and \
                len(self.history_checkpoints) > self.max_checkpoint_num:
            history_checkpoints = sorted(
                self.history_checkpoints, key=extract_metric_from_filename)
            self.history_checkpoints.clear()
            for i, checkpoint_path_prefix in enumerate(history_checkpoints):
                if i < self.max_checkpoint_num:
                    self.history_checkpoints.add(checkpoint_path_prefix)
                else:
                    self.logger.info(
                        f'deleting checkpoint: {checkpoint_path_prefix}')
                    self.remove_checkpoints(
                        trainer, checkpoint_path_prefix=checkpoint_path_prefix)

    def state_dict(self):
        return {
            'best_metric': self._best_metric,
        }

    def load_state_dict(self, state_dict):
        if state_dict is not None and len(state_dict) > 0:
            self._best_metric = state_dict.get('best_metric')
        else:
            self.logger.warning(
                'The state_dict is not available, the best metric value will be affected.'
            )

    def after_run(self, trainer):
        if self.restore_best:
            # If restore_best is True, will call the LoadCheckpointHook to load the best checkpoint
            # for later evaluation or prediction.
            LoadCheckpointHook.load_checkpoint(self._best_ckpt_file, trainer)


@HOOKS.register_module(module_name=Hooks.LoadCheckpointHook)
class LoadCheckpointHook(Hook):
    """Load a checkpoint file at the beginning of training or evaluating.

    This hook does not need to be configured or saved in the config file.
    User should use it by:
    >>> trainer.train('some-checkpoint', load_all_state=True)
    or
    >>> trainer.evaluate('some-checkpoint')
    instead.

    Args:
        checkpoint_file (str): The checkpoint file to be loaded.
        load_all_state (bool): Load all states(optimizer, epoch, lr_scheduler, random_state, etc.) when loading old
            training state file or not. The model's state dict will only be loaded if False.
        strict (bool): If strict, any unmatched keys will cause an error.
    """

    PRIORITY = Priority.HIGH

    _should_save = False

    _TWO_PTH_FILE_VERSION = '1.3.1'

    def __init__(
        self,
        checkpoint_file=None,
        load_all_state=True,
        strict=False,
    ):
        self.checkpoint_file = checkpoint_file
        self.rng_state = None
        self.need_load_rng_state = False
        self.load_all_state = load_all_state
        self.strict = strict

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

        if self.checkpoint_file is not None:
            meta = self.load_checkpoint(self.checkpoint_file, trainer,
                                        self.load_all_state, self.strict)
            self.rng_state = meta.get('rng_state')
            self.need_load_rng_state = self.load_all_state

    def before_train_iter(self, trainer):
        if self.need_load_rng_state:
            if self.rng_state is not None:
                random.setstate(self.rng_state['random'])
                np.random.set_state(self.rng_state['numpy'])
                torch.random.set_rng_state(self.rng_state['cpu'])
                if torch.cuda.is_available():
                    torch.cuda.random.set_rng_state_all(self.rng_state['cuda'])
                self.need_load_rng_state = False
            else:
                self.logger.info(
                    'Random state cannot be found in checkpoint file, '
                    'this may cause a random data order or model initialization.'
                )

    @staticmethod
    def _restore_training_state(trainer, meta):
        trainer._epoch = meta.get('epoch', trainer._epoch)
        trainer._iter = meta.get('iter', trainer._iter)
        trainer._inner_iter = meta.get('inner_iter', trainer._inner_iter)

        i = 0
        for hook in trainer.hooks:
            if hasattr(hook, 'load_state_dict') and getattr(
                    hook, '_should_save', True):
                key = f'{hook.__class__}-{i}'
                if key in meta:
                    hook.load_state_dict(meta.get(key, {}))
                else:
                    trainer.logger.warning(
                        f'The state_dict of hook {hook.__class__} at index {i} is not found in the checkpoint file.'
                    )
                i += 1

    @classmethod
    def load_checkpoint(cls,
                        filename,
                        trainer,
                        load_all_state=True,
                        strict=False):
        """A static method to load checkpoint files.

        Args:
            filename(str): An absolute model bin file(pth or bin) or a dir path with a file prefix(like epoch_1).
            trainer(`EpochBasedTrainer`): The trainer instance.
            load_all_state(`bool`): Load all states including the trainer states.
            strict(`bool`): Load module state dict strictly.

        Returns:
            A dict containing the train states saved by `_create_training_state`
        """
        meta = cls().load_checkpoints(filename, trainer, load_all_state,
                                      strict)
        if load_all_state:
            cls._restore_training_state(trainer, meta)

        if meta is not None:
            _version = meta.get('modelscope')
            if _version is not None and version.parse(
                    _version) < version.parse(
                        LoadCheckpointHook._TWO_PTH_FILE_VERSION):
                trainer.logger.warning(
                    'The unique pth file is split into a model file and '
                    f'a trainer file since version {LoadCheckpointHook._TWO_PTH_FILE_VERSION},'
                    'consider re-training your model or '
                    'using a converting script to split the single pth file into two.'
                )
            trainer.logger.info(
                f'Checkpoint {filename} saving time: {meta.get("time")}, modelscope version: {_version}'
            )
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

    @Hook.overload_func(name='LoadCheckpointHook.load_checkpoints')
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
        _model_file, _train_state_file = _get_state_file_name(
            checkpoint_path_prefix)
        meta = {}
        if os.path.isfile(_train_state_file):
            meta = self.load_trainer_state(trainer, _train_state_file,
                                           load_all_state)
        else:
            print(f'No trainer state file {_train_state_file} found, skip.')
        self.load_model_state(trainer, _model_file, strict)
        return meta


def _get_state_file_name(checkpoint_path_prefix):
    """Get the default file name for state files.

    If the input is a checkpoint dir with prefix, this function will append suffix for both checkpoint files.
    If the input is an absolute file name, this function will return it as the model file name, and append
        suffix for the trainer file name.

    NOTE: a best checkpoint filename with float or int metric value inside
        will not be judged as having a extension file name. like: '/tmp/test/epoch_0_accuracy0.85'

    Args:
        checkpoint_path_prefix(`str`): The checkpoint dir with prefix or a model state file with extension file name.
            like: '/tmp/test/epoch_0'

    Returns:
          A tuple of model state file name and trainer state file name.
    """
    base, ext = os.path.splitext(checkpoint_path_prefix)
    if len(ext) == 0 or re.match(r'^\d+$', ext[1:]):
        return checkpoint_path_prefix + CheckpointHook.MODEL_STATE_SUFFIX, \
            checkpoint_path_prefix + CheckpointHook.TRAINER_STATE_SUFFIX
    else:
        return checkpoint_path_prefix, base + CheckpointHook.TRAINER_STATE_SUFFIX.split(
            '.')[0] + '.' + ext[1:]
