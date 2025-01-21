'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
from megatron_util import mpu


def _gather_tokens(input_, dim=0):
    """Gather tensors and concatenate them along a dimension"""

    input_ = input_.contiguous()
    # Size and dimension.
    rank = mpu.get_tensor_model_parallel_rank()

    tensor_list = [
        torch.empty_like(input_)
        for _ in range(mpu.get_tensor_model_parallel_world_size())
    ]
    tensor_list[rank] = input_
    torch.distributed.all_gather(
        tensor_list, input_, group=mpu.get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _drop_tokens(input_, dim=0):
    """Divide a tensor among the tensor parallel ranks"""
    total_chunks = mpu.get_tensor_model_parallel_world_size()
    this_chunk = mpu.get_tensor_model_parallel_rank()
    assert input_.shape[
        dim] % total_chunks == 0, f'input dimension {dim} ({input_.shape[dim]}) ' \
                                  f'is not divisible by tensor parallel world size ({total_chunks})'
    chunk_size = input_.shape[dim] // total_chunks

    return torch.narrow(input_, dim, this_chunk * chunk_size, chunk_size)


class _GatherTokens(torch.autograd.Function):
    """All gather tokens among the tensor parallel ranks"""

    @staticmethod
    def symbolic(graph, input_, dim):
        return _gather_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _gather_tokens(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_tokens(grad_output, ctx.dim), None


class _DropTokens(torch.autograd.Function):
    'Divide tokens equally among the tensor parallel ranks'

    @staticmethod
    def symbolic(graph, input_, dim):
        return _drop_tokens(input_, dim)

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _drop_tokens(input_, dim)

    @staticmethod
    def backward(ctx, input_):
        return _gather_tokens(input_, ctx.dim), None


def gather_tokens(input_, dim=0):
    if mpu is None or mpu.get_tensor_model_parallel_world_size() == 1:
        # no tensor parallelism for non-experts
        return input_
    return _GatherTokens.apply(input_, dim)


def drop_tokens(input_, dim=0):
    if mpu is None or mpu.get_tensor_model_parallel_world_size() == 1:
        # no tensor parallelism for non-experts
        return input_
    return _DropTokens.apply(input_, dim)
