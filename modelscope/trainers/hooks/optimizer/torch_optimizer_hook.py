# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from .base import OptimizerHook


@HOOKS.register_module(module_name=Hooks.TorchAMPOptimizerHook)
class TorchAMPOptimizerHook(OptimizerHook):
    """
    Fp16 optimizer, if torch version is less than 1.6.0,
    you must install apex (https://www.github.com/nvidia/apex) else use torch.cuda.amp by default

    Args:
        cumulative_iters (int): interval of gradients accumulation. Default: 1
        grad_clip (dict): Default None. Containing keys:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            More details please refer to `torch.nn.utils.clip_grad.clip_grad_norm_`
        loss_keys (str | list): keys list of loss
        loss_scale (float | dict): grade scale config. If loss_scale is a float,
            static loss scaling will be used with the specified scale.
            It can also be a dict containing arguments of GradScalar. For Pytorch >= 1.6,
            we use official torch.cuda.amp.GradScaler.
            please refer to: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler for the parameters.
    """

    def __init__(self,
                 cumulative_iters=1,
                 grad_clip=None,
                 loss_keys='loss',
                 loss_scale={}):

        super(TorchAMPOptimizerHook, self).__init__(
            grad_clip=grad_clip, loss_keys=loss_keys)
        self.cumulative_iters = cumulative_iters
        self._scale_update_param = None

        from torch.cuda import amp

        if isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.scaler = amp.GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.scaler = amp.GradScaler(**loss_scale)
        else:
            raise ValueError(
                '`loss_scale` type must be in [float, dict], but got {loss_scale}'
            )

    def before_run(self, trainer):
        logging.info('open fp16')
        trainer.optimizer.zero_grad()

        if hasattr(trainer.model, 'module'):
            self._ori_model_forward = trainer.model.module.forward
            self._model = trainer.model.module
        else:
            self._ori_model_forward = trainer.model.forward
            self._model = trainer.model

        self.ori_model_forward = trainer.model.forward

    def before_train_iter(self, trainer):
        from torch.cuda import amp
        setattr(self._model, 'forward', amp.autocast()(self._model.forward))

    def after_train_iter(self, trainer):
        for k in self.loss_keys:
            trainer.train_outputs[k] /= self.cumulative_iters

        for k in self.loss_keys:
            self.scaler.scale(trainer.train_outputs[k]).backward()

        if self.every_n_iters(trainer, self.cumulative_iters):
            self.scaler.unscale_(trainer.optimizer)
            if self.grad_clip is not None:
                self.clip_grads(trainer.model.parameters(), **self.grad_clip)

            self.scaler.step(trainer.optimizer)
            self.scaler.update(self._scale_update_param)
            trainer.optimizer.zero_grad()

        setattr(self._model, 'forward', self._ori_model_forward)
