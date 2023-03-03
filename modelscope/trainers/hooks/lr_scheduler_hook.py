# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import Hooks
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.utils.constant import LogKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module(module_name=Hooks.LrSchedulerHook)
class LrSchedulerHook(Hook):
    """Lr scheduler.

    Args:
        by_epoch (bool): Whether lr changes by epoch
        warmup (dict): warm up config
    """
    PRIORITY = Priority.LOW

    def __init__(self, by_epoch=True, warmup=None, **kwargs) -> None:
        super().__init__()
        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_lr_scheduler = None

    def before_run(self, trainer):
        self.initialize_lr_scheduler(trainer)
        if self.warmup is not None:
            assert isinstance(self.warmup, dict) and 'type' in self.warmup
            self.warmup_lr_scheduler = build_lr_scheduler(
                cfg=self.warmup,
                default_args={'base_scheduler': trainer.lr_scheduler})

    @Hook.overload_func(name='LrSchedulerHook.initialize_lr_scheduler')
    def initialize_lr_scheduler(self, trainer):
        """Initialize the lr scheduler.

        This is a strategic function which can be registered by other hook's function.
        """
        pass

    @Hook.overload_func(name='LrSchedulerHook.step')
    def step(self, trainer):
        """Do lr scheduler's step.

        This is a strategic function which can be registered by other hook's function.
        """
        if self.warmup_lr_scheduler is not None:
            self.warmup_lr_scheduler.step()
        else:
            trainer.lr_scheduler.step()

    def get_current_lr(self, trainer):
        import torch

        if isinstance(trainer.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in trainer.optimizer.param_groups]
        elif isinstance(trainer.optimizer, dict):
            lr = dict()
            for name, optim in trainer.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def after_train_iter(self, trainer):
        if not self.by_epoch and trainer.iter >= getattr(
                trainer, 'cumulative_iters', 1) - 1:
            self.step(trainer)
        trainer.log_buffer.output[LogKeys.LR] = self._get_log_lr(trainer)

    def before_train_epoch(self, trainer):
        trainer.log_buffer.output[LogKeys.LR] = self._get_log_lr(trainer)

    def after_train_epoch(self, trainer):
        if self.by_epoch:
            self.step(trainer)

    def _get_log_lr(self, trainer):
        cur_lr = self.get_current_lr(trainer)
        # only record lr of the first param group
        if isinstance(cur_lr, list):
            lr = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            lr = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                lr.update({k: lr_[0]})

        return lr


@HOOKS.register_module(module_name=Hooks.PlateauLrSchedulerHook)
class PlateauLrSchedulerHook(Hook):
    """Lr scheduler hook for `ReduceLROnPlateau`.

    Args:
        metric_key (str): Metric key returned from `trainer.metric_values`,
            get the value of metric key and pass it to `ReduceLROnPlateau.step`.
    """
    PRIORITY = Priority.LOW  # should be after EvaluationHook

    def __init__(self, metric_key, **kwargs):
        self.metric_key = metric_key

    def register_strategy(self):
        Hook.overload(name='LrSchedulerHook.step', function=self.step)

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

    def step(self, trainer):
        # adapt to evaluation intervel is greater than 1
        if trainer.metric_values is None:
            if is_master():
                self.logger.warning(
                    f'Current epoch {trainer.epoch} has no evaluation metric values, skip lr_scheduler.step() !'
                )
            return

        metrics = trainer.metric_values[self.metric_key]
        lr_scheduler_hook = trainer.get_hook(LrSchedulerHook)[0]
        if lr_scheduler_hook.by_epoch:
            if lr_scheduler_hook.warmup_lr_scheduler is not None:
                lr_scheduler_hook.warmup_lr_scheduler.step(metrics=metrics)
            else:
                trainer.lr_scheduler.step(metrics=metrics)


@HOOKS.register_module(module_name=Hooks.NoneLrSchedulerHook)
class NoneLrSchedulerHook(LrSchedulerHook):

    PRIORITY = Priority.LOW  # should be after EvaluationHook

    def __init__(self, by_epoch=True, warmup=None) -> None:
        super().__init__(by_epoch=by_epoch, warmup=warmup)

    def before_run(self, trainer):
        return

    def after_train_epoch(self, trainer):
        return
