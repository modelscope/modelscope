# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random

import numpy as np
import torch

from modelscope import __version__
from modelscope.metainfo import Hooks
from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import get_dist_info, is_master
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
        save_last (bool): Whether to save the last checkpoint. Default: True.
        checkpoint_file (str): The checkpoint file to be loaded.
    """

    PRIORITY = Priority.LOW

    def __init__(self,
                 interval=0,
                 by_epoch=True,
                 save_optimizer=True,
                 save_dir=None,
                 save_last=True,
                 checkpoint_file=None):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_dir = save_dir
        self.checkpoint_file = checkpoint_file
        self.save_last = save_last
        self.rng_state = None
        self.need_load_rng_state = False

    def before_run(self, trainer):
        if not self.save_dir:
            self.save_dir = trainer.work_dir

        if not os.path.exists(self.save_dir) and is_master():
            os.makedirs(self.save_dir)

        if not hasattr(trainer, 'logger'):
            self.logger = get_logger(__name__)
        else:
            self.logger = trainer.logger

        if is_master():
            self.logger.info(f'Checkpoints will be saved to {self.save_dir}')

        if self.checkpoint_file is not None and os.path.isfile(
                self.checkpoint_file):
            meta = self.load_checkpoint(self.checkpoint_file, trainer)
            self.rng_state = meta.get('rng_state')
            self.need_load_rng_state = True

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
                self.logger.warn(
                    'Random state cannot be found in checkpoint file, '
                    'this may cause a random data order or model initialization.'
                )

    def after_train_epoch(self, trainer):
        if not self.by_epoch:
            return

        if self._should_save(trainer):
            if is_master():
                self.logger.info(
                    f'Saving checkpoint at {trainer.epoch + 1} epoch')
                self._save_checkpoint(trainer)

    @classmethod
    def load_checkpoint(cls, filename, trainer):
        from modelscope.trainers.parallel.utils import is_parallel
        if is_parallel(trainer.model):
            model = trainer.model.module
        else:
            model = trainer.model
        meta = load_checkpoint(filename, model,
                               getattr(trainer, 'optimizer', None),
                               getattr(trainer, 'lr_scheduler', None))
        trainer._epoch = meta.get('epoch', trainer._epoch)
        trainer._iter = meta.get('iter', trainer._iter)
        trainer._inner_iter = meta.get('inner_iter', trainer._inner_iter)

        for i, hook in enumerate(trainer.hooks):
            # hook: Hook
            key = f'{hook.__class__}-{i}'
            if key in meta and hasattr(hook, 'load_state_dict'):
                hook.load_state_dict(meta.get(key, {}))
            else:
                trainer.logger.warn(
                    f'The state_dict of hook {hook.__class__} at index {i} is not found in the checkpoint file.'
                )

        version = meta.get('modelscope')
        if version != __version__:
            trainer.logger.warn(
                f'The modelscope version of loaded checkpoint does not match the runtime version. '
                f'The saved version: {version}, runtime version: {__version__}'
            )
        trainer.logger.info(
            f'Checkpoint {filename} saving time: {meta.get("time")}')
        return meta

    def _save_checkpoint(self, trainer):
        if self.by_epoch:
            cur_save_name = os.path.join(
                self.save_dir, f'{LogKeys.EPOCH}_{trainer.epoch + 1}.pth')
        else:
            cur_save_name = os.path.join(
                self.save_dir, f'{LogKeys.ITER}_{trainer.iter + 1}.pth')

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
        for i, hook in enumerate(trainer.hooks):
            if hasattr(hook, 'state_dict'):
                meta[f'{hook.__class__}-{i}'] = hook.state_dict()

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

    def _save_pretrained(self, trainer):
        output_dir = os.path.join(self.save_dir, ModelFile.TRAIN_OUTPUT_DIR)
        from modelscope.trainers.parallel.utils import is_parallel

        if is_parallel(trainer.model):
            model = trainer.model.module
        else:
            model = trainer.model

        config = trainer.cfg.to_dict()
        # override pipeline by tasks name after finetune done,
        # avoid case like fill mask pipeline with a text cls task
        config['pipeline'] = {'type': config['task']}

        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(
                output_dir,
                ModelFile.TORCH_MODEL_BIN_FILE,
                save_function=save_checkpoint,
                config=config,
                with_meta=False)

    def after_train_iter(self, trainer):
        if self.by_epoch:
            return

        if self._should_save(trainer):
            if is_master():
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
    """Save best checkpoints hook.
    Args:
        metric_key (str): Metric key to compare rule for best score.
        rule (str): Comparison rule for best score.
            Support "max" and "min". If rule is "max", the checkpoint at the maximum `metric_key`
            will be saved, If rule is "min", the checkpoint at the minimum `metric_key` will be saved.
        by_epoch (bool): Save best checkpoints by epoch or by iteration.
        save_optimizer (bool): Whether to save optimizer state dict.  Default: True.
        save_dir (str): Output directory to save best checkpoint.
        restore_best (bool): Whether to restore the best checkpoint after training.
    """

    PRIORITY = Priority.LOW
    rule_map = {'max': lambda x, y: x > y, 'min': lambda x, y: x < y}

    def __init__(self,
                 metric_key,
                 rule='max',
                 by_epoch=True,
                 save_optimizer=True,
                 save_dir=None,
                 save_file_name=None,
                 restore_best=False,
                 interval=0):
        assert rule in ['max', 'min'], 'Only support "max" or "min" rule now.'
        super().__init__(
            interval=interval,
            by_epoch=by_epoch,
            save_optimizer=save_optimizer,
            save_dir=save_dir,
        )
        self.metric_key = metric_key
        self.rule = rule
        self._best_metric = None
        self._best_ckpt_file = None
        self.save_file_name = save_file_name
        self.restore_best = restore_best

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

        meta = {
            'epoch': trainer.epoch,
            'iter': trainer.iter + 1,
            'inner_iter': trainer.inner_iter + 1,
            'rng_state': self.rng_state,
        }
        for i, hook in enumerate(trainer.hooks):
            meta[f'{hook.__class__}-{i}'] = hook.state_dict()

        if os.path.isfile(cur_save_name):
            os.remove(cur_save_name)
        save_checkpoint(trainer.model, cur_save_name, trainer.optimizer,
                        trainer.lr_scheduler, meta)
        self._best_ckpt_file = cur_save_name
        self._save_pretrained(trainer)

    def state_dict(self):
        return {
            'best_metric': self._best_metric,
        }

    def load_state_dict(self, state_dict):
        if state_dict is not None and len(state_dict) > 0:
            self._best_metric = state_dict.get('best_metric')
        else:
            self.logger.warn(
                'The state_dict is not available, the best metric value will be affected.'
            )

    def after_run(self, trainer):
        if self.restore_best:
            self.load_checkpoint(self._best_ckpt_file, trainer)
