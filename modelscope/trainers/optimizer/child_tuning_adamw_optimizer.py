# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import types
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Optimizer

from modelscope.utils.logger import get_logger
from .builder import OPTIMIZERS, default_group

logger = get_logger()

__all__ = ['calculate_fisher', 'ChildTuningAdamW']


def calculate_fisher(model: torch.nn.Module,
                     data_loader,
                     forward_step,
                     reserve_p,
                     grad_clip=None):

    gradient_mask = dict()
    model.train()
    for name, params in model.named_parameters():
        if 'layer' in name:
            gradient_mask[params] = params.new_zeros(params.size())

    iters = len(data_loader)
    for inputs in data_loader:
        loss = forward_step(model, inputs)
        loss.backward()
        for name, params in model.named_parameters():
            if 'layer' in name:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, **grad_clip)
                gradient_mask[params] += (params.grad**2) / iters
        model.zero_grad()

    logger.info('Calculate Fisher Information...')

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1 - reserve_p) * 100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar
    print('Polar => {}'.format(polar))

    # TODO: pytorch: torch.kthvalue

    return gradient_mask


@OPTIMIZERS.register_module(
    group_key=default_group, module_name='ChildTuningAdamW')
class ChildTuningAdamW(Optimizer):

    def __init__(self,
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.0,
                 correct_bias: bool = True,
                 reserve_p=1.0,
                 mode=None):
        if lr < 0.0:
            raise ValueError(
                'Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter: {} - should be in [0.0, 1.0['.format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter: {} - should be in [0.0, 1.0['.format(
                    betas[1]))
        if not 0.0 <= eps:
            raise ValueError(
                'Invalid epsilon value: {} - should be >= 0.0'.format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )

                # ChildTuning code
                if self.mode is not None:
                    if self.mode == 'ChildTuning-D':
                        if p in self.gradient_mask:
                            grad *= self.gradient_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = Bernoulli(
                            grad.new_full(
                                size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1**state['step']
                    bias_correction2 = 1.0 - beta2**state['step']
                    step_size = step_size * math.sqrt(
                        bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss
