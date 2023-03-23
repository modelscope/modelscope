# Copyright (c) 2022 Zhipu.AI
import ctypes
from typing import List

import pkg_resources
import torch
from cpm_kernels.kernels.base import (KernelFunction, LazyKernelCModule,
                                      round_up)

RESOURCE_PACKAGE_NAME = __name__


class Kernel:

    def __init__(self, filename: str, function_names: List[str]):
        filename = filename + '.fatbin'
        if not pkg_resources.resource_exists(RESOURCE_PACKAGE_NAME, filename):
            raise RuntimeError('File `%s` not found in `%s`' %
                               (filename, RESOURCE_PACKAGE_NAME))
        self.filename = filename
        self.code = pkg_resources.resource_string(RESOURCE_PACKAGE_NAME,
                                                  filename)
        self._function_names = function_names
        self._cmodule = LazyKernelCModule(self.code)

        for name in self._function_names:
            setattr(self, name, KernelFunction(self._cmodule, name))


kernels = Kernel(
    'quantization',
    [
        'int4WeightCompression',
        'int4WeightExtractionFloat',
        'int4WeightExtractionHalf',
        'int8WeightExtractionFloat',
        'int8WeightExtractionHalf',
    ],
)


def compress_int4_weight(weight: torch.Tensor):  # (n, m)
    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device='cuda')
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        kernels.int4WeightCompression(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m)
            ],
        )
        return out


def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor,
                           source_bit_width: int):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, 'Unsupported bit-width'

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(
            n, m * (8 // source_bit_width), dtype=torch.half, device='cuda')
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


if __name__ == '__main__':
    weight = torch.randn(4, 32).to(torch.int8).cuda()
    scale = torch.ones(weight.size(0)).to(torch.half).cuda()

    print(weight)
    b = compress_int4_weight(weight)
    print(b)

    a = extract_weight_to_half(b, scale, source_bit_width=4)
    print(a)
