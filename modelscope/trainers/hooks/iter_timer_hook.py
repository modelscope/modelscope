# Copyright (c) Alibaba, Inc. and its affiliates.
import time

from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module()
class IterTimerHook(Hook):
    PRIORITY = Priority.LOW

    def before_epoch(self, trainer):
        self.start_time = time.time()

    def before_iter(self, trainer):
        trainer.log_buffer.update(
            {'data_load_time': time.time() - self.start_time})

    def after_iter(self, trainer):
        trainer.log_buffer.update({'time': time.time() - self.start_time})
        self.start_time = time.time()
