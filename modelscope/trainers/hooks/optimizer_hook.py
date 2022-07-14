# Copyright (c) Alibaba, Inc. and its affiliates.
from torch.nn.utils import clip_grad

from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module()
class OptimizerHook(Hook):

    PRIORITY = Priority.ABOVE_NORMAL

    def __init__(self, grad_clip=None, loss_keys='loss') -> None:
        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        assert isinstance(loss_keys, (tuple, list))
        self.loss_keys = loss_keys
        self.grad_clip = grad_clip

    def clip_grads(self, params, **clip_args):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **clip_args)

    def after_train_iter(self, trainer):
        trainer.optimizer.zero_grad()

        for k in self.loss_keys:
            trainer.train_outputs[k].backward()

        clip_args = self.grad_clip
        if clip_args is not None:
            self.clip_grads(trainer.model.parameters(), **clip_args)

        trainer.optimizer.step()
