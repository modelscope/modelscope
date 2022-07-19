# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from torch.nn.utils import clip_grad

from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module()
class OptimizerHook(Hook):
    """Optimizer hook

    Args:
        cumulative_iters (int): interval of gradients accumulation. Default: 1
        grad_clip (dict): Default None. Containing keys:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            More details please refer to `torch.nn.utils.clip_grad.clip_grad_norm_`
        loss_keys (str | list): keys list of loss
    """

    PRIORITY = Priority.ABOVE_NORMAL

    def __init__(self,
                 cumulative_iters=1,
                 grad_clip=None,
                 loss_keys='loss') -> None:
        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        assert isinstance(loss_keys, (tuple, list))
        self.loss_keys = loss_keys
        self.cumulative_iters = cumulative_iters
        self.grad_clip = grad_clip

    def clip_grads(self, params, **clip_args):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **clip_args)

    def before_run(self, trainer):
        trainer.optimizer.zero_grad()

    def after_train_iter(self, trainer):
        for k in self.loss_keys:
            trainer.train_outputs[k] /= self.cumulative_iters
            trainer.train_outputs[k].backward()

        if self.every_n_iters(trainer, self.cumulative_iters):
            if self.grad_clip is not None:
                self.clip_grads(trainer.model.parameters(), **self.grad_clip)

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()


@HOOKS.register_module()
class TorchAMPOptimizerHook(OptimizerHook):
    """Fp16 optimizer, if torch version is less than 1.6.0,
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


@HOOKS.register_module()
class ApexAMPOptimizerHook(OptimizerHook):
    """Fp16 optimizer, if torch version is less than 1.6.0,
    you must install apex (https://www.github.com/nvidia/apex) else use torch.cuda.amp by default
    Args:
        cumulative_iters (int): interval of gradients accumulation. Default: 1
        grad_clip (dict): Default None. Containing keys:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            More details please refer to `torch.nn.utils.clip_grad.clip_grad_norm_`
        loss_keys (str | list): keys list of loss
        opt_level (str): "O0" and "O3" are not true mixed precision,
            but they are useful for establishing accuracy and speed baselines, respectively.
            "O1" and "O2" are different implementations of mixed precision.
            Try both, and see what gives the best speedup and accuracy for your model.
    """

    def __init__(self,
                 cumulative_iters=1,
                 grad_clip=None,
                 loss_keys='loss',
                 opt_level='O1'):

        super(ApexAMPOptimizerHook, self).__init__(
            grad_clip=grad_clip, loss_keys=loss_keys)
        self.cumulative_iters = cumulative_iters
        self.opt_level = opt_level

        try:
            from apex import amp
        except ImportError:
            raise ValueError(
                'apex not installed, please install apex from https://www.github.com/nvidia/apex.'
            )

    def before_run(self, trainer):
        from apex import amp

        logging.info('open fp16')
        # TODO: fix it should initialze amp with model not wrapper by DDP or DP
        if hasattr(trainer.model, 'module'):
            trainer.model, trainer.optimizer = amp.initialize(
                trainer.model.module,
                trainer.optimizer,
                opt_level=self.opt_level)
        else:
            trainer.model, trainer.optimizer = amp.initialize(
                trainer.model, trainer.optimizer, opt_level=self.opt_level)

        trainer.optimizer.zero_grad()

    def after_train_iter(self, trainer):
        for k in self.loss_keys:
            trainer.train_outputs[k] /= self.cumulative_iters

        from apex import amp
        for k in self.loss_keys:
            with amp.scale_loss(trainer.train_outputs[k],
                                trainer.optimizer) as scaled_loss:
                scaled_loss.backward()

        if self.every_n_iters(trainer, self.cumulative_iters):
            if self.grad_clip is not None:
                self.clip_grads(trainer.model.parameters(), **self.grad_clip)

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
