# Copyright (c) Alibaba, Inc. and its affiliates.

from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler


class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [
            base_lr * ((self.max_lr / base_lr)**(
                self.last_epoch /  # noqa W504
                (self.max_steps - 1))) for base_lr in self.base_lrs
        ]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps**0.5 * min(
            last_epoch**(-0.5), last_epoch * self.warmup_steps**(-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
