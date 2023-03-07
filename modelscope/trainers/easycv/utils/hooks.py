# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.trainers.hooks import HOOKS, Priority
from modelscope.trainers.hooks.lr_scheduler_hook import LrSchedulerHook
from modelscope.utils.constant import LogKeys


@HOOKS.register_module(module_name='AddLrLogHook')
class AddLrLogHook(LrSchedulerHook):
    """For EasyCV to adapt to ModelScope, the lr log of EasyCV is added in the trainer,
    but the trainer of ModelScope does not and it is added in the lr scheduler hook.
    But The lr scheduler hook used by EasyCV is the hook of mmcv, and there is no lr log.
    It will be deleted in the future.
    """
    PRIORITY = Priority.NORMAL

    def __init__(self):
        pass

    def before_run(self, trainer):
        pass

    def after_train_iter(self, trainer):
        trainer.log_buffer.output[LogKeys.LR] = self._get_log_lr(trainer)

    def before_train_epoch(self, trainer):
        trainer.log_buffer.output[LogKeys.LR] = self._get_log_lr(trainer)

    def after_train_epoch(self, trainer):
        pass
