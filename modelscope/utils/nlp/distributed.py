# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
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

import math

import torch
import torch.distributed as dist
from megatron_util import mpu
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.autograd import Variable
from torch.nn.modules import Module


def normal_init_method(mean, std):

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


def scaled_init_method(mean, std, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        self.module = module
        self.data_parallel_group = mpu.get_data_parallel_group()
        src_rank = mpu.get_tensor_model_parallel_rank()
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, src_rank, group=self.data_parallel_group)

        def allreduce_params(reduce_after=True,
                             no_scale=False,
                             fp32_allreduce=False):
            if (self.needs_reduction):
                self.needs_reduction = False
                buckets = {}
                for name, param in self.module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = (param.data.type())
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if self.warn_on_half:
                    if torch.cuda.HalfTensor in buckets:
                        print(
                            'WARNING: gloo dist backend for half parameters may be extremely slow.',
                            'It is recommended to use the NCCL backend in this case.'
                        )
                        self.warn_on_half = False
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    if fp32_allreduce:
                        coalesced = coalesced.float()
                    if not no_scale and not reduce_after:
                        coalesced /= dist.get_world_size(
                            group=self.data_parallel_group)
                    dist.all_reduce(coalesced, group=self.data_parallel_group)
                    torch.cuda.synchronize()
                    if not no_scale and reduce_after:
                        coalesced /= dist.get_world_size(
                            group=self.data_parallel_group)
                    for buf, synced in zip(
                            grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        self.hook_handles = []
        self.hooks = []
        for param in list(self.module.parameters()):

            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)

        self.allreduce_params = allreduce_params

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)

        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
