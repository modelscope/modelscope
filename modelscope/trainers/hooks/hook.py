# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import wraps

from modelscope.utils.constant import TrainerStages
from modelscope.utils.import_utils import is_method_overridden
from .priority import Priority


class Hook:
    """
    The Hook base class of any modelscope trainer. You can build your own hook inherited from this class.
    """

    stages = (TrainerStages.after_init, TrainerStages.before_run,
              TrainerStages.before_val, TrainerStages.before_train_epoch,
              TrainerStages.before_train_iter, TrainerStages.after_train_iter,
              TrainerStages.after_train_epoch, TrainerStages.before_val_epoch,
              TrainerStages.before_val_iter, TrainerStages.after_val_iter,
              TrainerStages.after_val_epoch, TrainerStages.after_run,
              TrainerStages.after_val)

    PRIORITY = Priority.NORMAL

    def after_init(self, trainer):
        """
        Will be called at the end of the trainer's `__init__` method
        """
        pass

    def before_run(self, trainer):
        """
        Will be called before trainer loop begins.
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        pass

    def after_run(self, trainer):
        """
        Will be called after trainer loop end.
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        pass

    def before_val(self, trainer):
        """
        Will be called before eval loop begins.
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        pass

    def after_val(self, trainer):
        """
        Will be called after eval loop end.
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        pass

    def before_epoch(self, trainer):
        """
        Will be called before every epoch begins.
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        pass

    def after_epoch(self, trainer):
        """
        Will be called after every epoch ends.
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        pass

    def before_iter(self, trainer):
        """
        Will be called before every loop begins.
        Args:
            trainer: The trainer instance.

        Returns: None
        """
        pass

    def after_iter(self, trainer):
        """
        Will be called after every loop ends.
        Args:
            trainer: The trainer instance.

        Returns: None
        """
        pass

    def before_train_epoch(self, trainer):
        """
        Will be called before every train epoch begins. Default call ``self.before_epoch``
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        self.before_epoch(trainer)

    def before_val_epoch(self, trainer):
        """
        Will be called before every validation epoch begins. Default call ``self.before_epoch``
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        self.before_epoch(trainer)

    def after_train_epoch(self, trainer):
        """
        Will be called after every train epoch ends. Default call ``self.after_epoch``
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        self.after_epoch(trainer)

    def after_val_epoch(self, trainer):
        """
        Will be called after every validation epoch ends. Default call ``self.after_epoch``
        Args:
            trainer: The trainer instance.

        Returns: None

        """
        self.after_epoch(trainer)

    def before_train_iter(self, trainer):
        """
        Will be called before every train loop begins. Default call ``self.before_iter``
        Args:
            trainer: The trainer instance.

        Returns: None
        """
        self.before_iter(trainer)

    def before_val_iter(self, trainer):
        """
        Will be called before every validation loop begins. Default call ``self.before_iter``
        Args:
            trainer: The trainer instance.

        Returns: None
        """
        self.before_iter(trainer)

    def after_train_iter(self, trainer):
        """
        Will be called after every train loop ends. Default call ``self.after_iter``
        Args:
            trainer: The trainer instance.

        Returns: None
        """
        self.after_iter(trainer)

    def after_val_iter(self, trainer):
        """
        Will be called after every validation loop ends. Default call ``self.after_iter``
        Args:
            trainer: The trainer instance.

        Returns: None
        """
        self.after_iter(trainer)

    @staticmethod
    def every_n_epochs(trainer, n):
        """
        Whether to reach every ``n`` epochs
        Returns: bool
        """
        return (trainer.epoch + 1) % n == 0 if n > 0 else False

    @staticmethod
    def every_n_inner_iters(runner, n):
        """
        Whether to reach every ``n`` iterations at every epoch
        Returns: bool
        """
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    @staticmethod
    def every_n_iters(trainer, n):
        """
        Whether to reach every ``n`` iterations
        Returns: bool
        """
        return (trainer.iter + 1) % n == 0 if n > 0 else False

    @staticmethod
    def end_of_epoch(trainer):
        """
        Whether to reach the end of every epoch
        Returns: bool
        """
        return trainer.inner_iter + 1 == trainer.iters_per_epoch

    @staticmethod
    def is_last_epoch(trainer):
        """
        Whether to reach the last epoch
        Returns: bool
        """
        return trainer.epoch + 1 == trainer.max_epochs

    @staticmethod
    def is_last_iter(trainer):
        """
        Whether to reach the last iteration in the entire training process
        Returns: bool
        """
        return trainer.iter + 1 == trainer.max_iters

    def get_triggered_stages(self):
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)

        return [stage for stage in Hook.stages if stage in trigger_stages]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
