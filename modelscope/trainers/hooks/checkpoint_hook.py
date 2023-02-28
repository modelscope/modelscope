# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
import os
import random

import numpy as np
import torch

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

        if not os.path.exists(self.save_dir) and is_master():
            os.makedirs(self.save_dir)

        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

        if is_master():
            self.logger.info(f'Checkpoints will be saved to {self.save_dir}')

    def after_train_epoch(self, trainer):
        if not self.by_epoch:
            return

        if self._should_save(trainer):
            if is_master() or trainer.cfg.model.get('model_parallel_size',
                                                    1) != 1:
                self.logger.info(
                    f'Saving checkpoint at {trainer.epoch + 1} epoch')
                self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer):
        if self.by_epoch:
            cur_save_name = os.path.join(
                self.save_dir, f'{LogKeys.EPOCH}_{trainer.epoch + 1}.pth')
        else:
            cur_save_name = os.path.join(
                self.save_dir, f'{LogKeys.ITER}_{trainer.iter + 1}.pth')
        cur_save_name = extend_save_name_for_parallel(cur_save_name)

        self.rng_state = {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'cpu': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
        }
        meta = {
            'epoch': trainer.epoch,
            'iter': trainer.iter + 1,
            'inner_iter': trainer.inner_iter + 1,
            'rng_state': self.rng_state,
        }

        i = 0
        for hook in trainer.hooks:
            if hasattr(hook, 'state_dict') and getattr(hook, '_should_save',
                                                       True):
                meta[f'{hook.__class__}-{i}'] = hook.state_dict()
                i += 1

        save_checkpoint(
            trainer.model,
            cur_save_name,
            trainer.optimizer,
            trainer.lr_scheduler,
            meta=meta)
        if (self.is_last_epoch(trainer)
                and self.by_epoch) or (self.is_last_iter(trainer)
                                       and not self.by_epoch):
            self._save_pretrained(trainer)

        self.history_checkpoints.append(cur_save_name)
        self.remove_obsolete_checkpoints()

    def remove_obsolete_checkpoints(self):
        if self.max_checkpoint_num is not None and \
                len(self.history_checkpoints) > self.max_checkpoint_num:
            history_checkpoints = [ckpt for ckpt in self.history_checkpoints]
            self.history_checkpoints.clear()
            for i, ckpt_file in enumerate(history_checkpoints):
                if i < len(history_checkpoints) - self.max_checkpoint_num:
                    if os.path.isfile(ckpt_file):
                        os.remove(ckpt_file)
                else:
                    self.history_checkpoints.append(ckpt_file)

    def _save_pretrained(self, trainer):
        output_dir = os.path.join(self.save_dir, self.output_sub_dir)
        from modelscope.trainers.parallel.utils import is_parallel

        if is_parallel(trainer.model):
            model = trainer.model.module
        else:
            model = trainer.model

        config = trainer.cfg.to_dict()
        # override pipeline by tasks name after finetune done,
        # avoid case like fill mask pipeline with a text cls task
        if config['task'] in [
                getattr(Pipelines, attr) for attr in dir(Pipelines)
                if not attr.startswith('__')
        ]:
            # TODO a temp fix to avoid pipeline_name and task mismatch
            config['pipeline'] = {'type': config['task']}

        # remove parallel module that is not JSON serializable
        if 'parallel' in config and 'module' in config['parallel']:
            del config['parallel']['module']

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
            # Now support two binary files: pytorch_model.bin and pytorch_model.pt
            default_bin_file = ModelFile.TORCH_MODEL_BIN_FILE
            if hasattr(
                    model,
                    'model_dir') and ModelFile.TORCH_MODEL_FILE in os.listdir(
                        model.model_dir):
                default_bin_file = ModelFile.TORCH_MODEL_FILE
            model.save_pretrained(
                output_dir,
                default_bin_file,
                save_function=save_checkpoint,
                config=save_config_fn.config,
                save_config_function=save_config_fn,
                with_meta=False)
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

    def after_train_iter(self, trainer):
        if self.by_epoch:
            return

        if self._should_save(trainer):
            if is_master() or trainer.cfg.model.get('model_parallel_size',
                                                    1) != 1:
                self.logger.info(
                    f'Saving checkpoint at {trainer.iter + 1} iterations')
                self._save_checkpoint(trainer)

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
        cur_save_name = self.save_file_name
        if cur_save_name is None:
            if self.by_epoch:
                cur_save_name = os.path.join(
                    self.save_dir,
                    f'best_{LogKeys.EPOCH}{trainer.epoch + 1}_{self.metric_key}{self._best_metric}.pth'
                )
            else:
                cur_save_name = os.path.join(
                    self.save_dir,
                    f'best_{LogKeys.ITER}{trainer.iter + 1}_{self.metric_key}{self._best_metric}.pth'
                )
        else:
            if '.' not in cur_save_name:
                cur_save_name = f'{cur_save_name}.pth'
            cur_save_name = os.path.join(self.save_dir, cur_save_name)
        cur_save_name = extend_save_name_for_parallel(cur_save_name)

        meta = {
            'epoch': trainer.epoch,
            'iter': trainer.iter + 1,
            'inner_iter': trainer.inner_iter + 1,
            'rng_state': self.rng_state,
        }

        i = 0
        for hook in trainer.hooks:
            if hasattr(hook, 'state_dict') and getattr(hook, '_should_save',
                                                       True):
                meta[f'{hook.__class__}-{i}'] = hook.state_dict()
                i += 1

        if os.path.isfile(cur_save_name):
            os.remove(cur_save_name)
        save_checkpoint(trainer.model, cur_save_name, trainer.optimizer,
                        trainer.lr_scheduler, meta)
        self._best_ckpt_file = cur_save_name
        self._save_pretrained(trainer)
        self.history_checkpoints.add(cur_save_name)
        self.remove_obsolete_checkpoints()

    def remove_obsolete_checkpoints(self):

        def extract_metric_from_filename(name1):
            metric1 = float('.'.join(
                name1.split(self.metric_key)[1].split('.')[:-1]))
            if self.rule == 'max':
                return -metric1
            else:
                return metric1

        if self.max_checkpoint_num is not None and \
                len(self.history_checkpoints) > self.max_checkpoint_num:
            history_checkpoints = sorted(
                self.history_checkpoints, key=extract_metric_from_filename)
            self.history_checkpoints.clear()
            for i, ckpt_file in enumerate(history_checkpoints):
                if i < self.max_checkpoint_num:
                    self.history_checkpoints.add(ckpt_file)
                elif os.path.isfile(ckpt_file):
                    os.remove(ckpt_file)

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
            if is_master():
                LoadCheckpointHook.load_checkpoint(self._best_ckpt_file,
                                                   trainer)


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
    """

    PRIORITY = Priority.HIGH

    _should_save = False

    def __init__(
        self,
        checkpoint_file=None,
        load_all_state=True,
    ):
        self.checkpoint_file = checkpoint_file
        self.rng_state = None
        self.need_load_rng_state = False
        self.load_all_state = load_all_state

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

        if self.checkpoint_file is not None and os.path.isfile(
                self.checkpoint_file):
            meta = self.load_checkpoint(self.checkpoint_file, trainer,
                                        self.load_all_state)
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
                self.logger.warning(
                    'Random state cannot be found in checkpoint file, '
                    'this may cause a random data order or model initialization.'
                )

    @classmethod
    def load_checkpoint(cls, filename, trainer, load_all_state=True):
        from modelscope.trainers.parallel.utils import is_parallel
        if is_parallel(trainer.model):
            model = trainer.model.module
        else:
            model = trainer.model
        meta = load_checkpoint(
            filename, model,
            getattr(trainer, 'optimizer', None) if load_all_state else None,
            getattr(trainer, 'lr_scheduler', None) if load_all_state else None)
        if load_all_state:
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

        version = meta.get('modelscope')
        if version != __version__:
            trainer.logger.warning(
                f'The modelscope version of loaded checkpoint does not match the runtime version. '
                f'The saved version: {version}, runtime version: {__version__}'
            )
        trainer.logger.info(
            f'Checkpoint {filename} saving time: {meta.get("time")}')
        return meta


def extend_save_name_for_parallel(cur_save_name: str) -> str:
    """Saving model parameters during tensor parallel training
    requires each process to save its own parameters,
    This function will try to get the local rank of the process
    and extend save name for multi-slice model.

    Args:
        cur_save_name (str): Original save name.

    Returns:
        str: Extended save name.
    """
    try:
        mpu = importlib.import_module('megatron_util.mpu')
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return cur_save_name
        mp_rank = mpu.get_tensor_model_parallel_rank()
        return cur_save_name.replace('.', '_mp_rank_{:02d}.'.format(mp_rank))
    except (ImportError, AssertionError):
        return cur_save_name
