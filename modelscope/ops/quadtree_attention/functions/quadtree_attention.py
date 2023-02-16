# Copyright (c) Alibaba, Inc. and its affiliates.

from pathlib import Path

import torch
from einops.einops import rearrange
from torch.autograd import Function
from torch.utils.cpp_extension import load

cur_dir = Path(__file__).parent.resolve()
score_computation_cuda = \
    load(name='score_computation_cuda', # noqa
        sources=[str(cur_dir / '../src/score_computation.cpp'), # noqa
                str(cur_dir / '../src/score_computation_kernal.cu')], # noqa
        extra_cflags=['-g'], extra_cuda_cflags=['-O2']) # noqa

value_aggregation_cuda = \
    load(name='value_aggregation_cuda', # noqa
        sources=[str(cur_dir / '../src/value_aggregation.cpp'), # noqa
                str(cur_dir / '../src/value_aggregation_kernel.cu')], # noqa
        extra_cflags=['-g'], extra_cuda_cflags=['-O2']) # noqa


class ScoreComputation(Function):

    @staticmethod
    def forward(ctx, query, key, index):
        x = score_computation_cuda.score_forward(query, key, index)
        ctx.save_for_backward(query, key, index)
        return x[0]

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, index = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        x = score_computation_cuda.score_backward(grad_output, input1, input2,
                                                  index)
        return x[0], x[1], None


score_computation_op = ScoreComputation.apply


class value_aggregation(Function):

    @staticmethod
    def forward(ctx, score, value, index):
        ctx.save_for_backward(score, value, index)
        f = score.shape[2]
        score = rearrange(
            score,
            'b n f K h -> b (n f) K h')  # [b, N, 4, 4K, H] -> [b, 4N, 4K, H]
        index = rearrange(
            index,
            'b n f K h -> b (n f) K h')  # [b, N, 4, 4K, H] -> [b, 4N, 4K, H]
        b, N, _, H = score.shape
        D = value.shape[-1]
        # value [b, M, H, D]
        output = score.new_zeros([b, N, H, D]).contiguous()  # b, 4N, H, D
        value_aggregation_cuda.value_aggregation_forward(
            score, value, index, output)
        output = rearrange(output, 'b (n f) h d -> b n f h d', f=f)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        score, value, index = ctx.saved_tensors
        f = score.shape[2]
        score = rearrange(score, 'b n f K h -> b (n f) K h')
        index = rearrange(index, 'b n f K h -> b (n f) K h')

        grad_output = grad_output.contiguous()

        grad_score = score.new_zeros(score.shape).contiguous()
        grad_value = value.new_zeros(value.shape).contiguous()

        value_aggregation_cuda.value_aggregation_backward(
            grad_output, score, value, index, grad_score, grad_value)
        grad_score = rearrange(grad_score, 'b (n f) K h -> b n f K h', f=f)
        return grad_score, grad_value, None


value_aggregation_op = value_aggregation.apply
