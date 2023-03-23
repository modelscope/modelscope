# Copyright (c) 2022 Zhipu.AI
import torch
from SwissArmyTransformer.mpu import (ColumnParallelLinear, RowParallelLinear,
                                      copy_to_model_parallel_region,
                                      gather_from_model_parallel_region,
                                      reduce_from_model_parallel_region,
                                      scatter_to_model_parallel_region)
from torch.nn.parameter import Parameter

from ..kernels import compress_int4_weight
from .functional import W8A16Linear


class QuantizedColumnParallelLinear(ColumnParallelLinear):

    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedColumnParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(
                shape[0],
                shape[1] * weight_bit_width // 8,
                dtype=torch.int8,
                device=kwargs['device'])
            self.weight_scale = torch.empty(
                shape[0],
                dtype=kwargs['params_dtype'],
                device=kwargs['device'])
        else:
            self.weight_scale = (
                weight.abs().max(dim=-1).values / (  # noqa
                    (2**(weight_bit_width - 1)) - 1)).half()  # noqa
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(
                torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(
            self.weight.to(kwargs['device']), requires_grad=False)
        self.weight_scale = Parameter(
            self.weight_scale.to(kwargs['device']), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight,
                                            self.weight_scale,
                                            self.weight_bit_width)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class QuantizedRowParallelLinear(RowParallelLinear):

    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedRowParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(
                shape[0],
                shape[1] * weight_bit_width // 8,
                dtype=torch.int8,
                device=kwargs['device'])
            self.weight_scale = torch.empty(
                shape[0],
                dtype=kwargs['params_dtype'],
                device=kwargs['device'])
        else:
            self.weight_scale = (
                weight.abs().max(dim=-1).values / (  # noqa
                    (2**(weight_bit_width - 1)) - 1)).half()  # noqa
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(
                torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(
            self.weight.to(kwargs['device']), requires_grad=False)
        self.weight_scale = Parameter(
            self.weight_scale.to(kwargs['device']), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight,
                                            self.weight_scale,
                                            self.weight_bit_width)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
