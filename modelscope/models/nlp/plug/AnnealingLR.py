# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch DataLoader for TFRecords"""

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', 'None']

    def __init__(self,
                 optimizer,
                 start_lr,
                 warmup_iter,
                 num_iters,
                 decay_style=None,
                 last_iter=-1):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self._step_count = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style,
                                                             str) else None
        self.step(self._step_count)
        if torch.distributed.get_rank() == 0:
            print('learning rate decaying', decay_style)

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.warmup_iter > 0 and self._step_count <= self.warmup_iter:
            return float(self.start_lr) * self._step_count / self.warmup_iter
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                return self.start_lr * ((
                    self.end_iter -  # noqa W504
                    (self._step_count - self.warmup_iter)) / self.end_iter)
            elif self.decay_style == self.DECAY_STYLES[1]:
                return self.start_lr / 2.0 * (
                    math.cos(math.pi * (self._step_count - self.warmup_iter)
                             / self.end_iter) + 1)
            elif self.decay_style == self.DECAY_STYLES[2]:
                # TODO: implement exponential decay
                return self.start_lr
            else:
                return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self._step_count + 1
        self._step_count = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
            'start_lr': self.start_lr,
            'warmup_iter': self.warmup_iter,
            '_step_count': self._step_count,
            'decay_style': self.decay_style,
            'end_iter': self.end_iter
        }
        return sd

    def load_state_dict(self, sd):
        self.start_lr = sd['start_lr']
        self.warmup_iter = sd['warmup_iter']
        self._step_count = sd['_step_count']
        self.end_iter = sd['end_iter']
        self.decay_style = sd['decay_style']
        self.step(self._step_count)
