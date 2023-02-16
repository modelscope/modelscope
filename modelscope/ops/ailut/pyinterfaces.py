import os.path as osp
from typing import Tuple

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load

try:
    from cudaops_ailut import (ailut_cbackward, ailut_cforward, lut_cbackward,
                               lut_cforward)
except ImportError:
    CUR_DIR = osp.abspath(osp.dirname(__file__))
    cudaops_ailut = load(
        name='cudaops_ailut',
        sources=[
            osp.join(CUR_DIR, 'Ailut', 'csrc/ailut_transform.cpp'),
            osp.join(CUR_DIR, 'Ailut', 'csrc/ailut_transform_cpu.cpp'),
            osp.join(CUR_DIR, 'Ailut', 'csrc/ailut_transform_cuda.cu')
        ],
        verbose=True)
    from cudaops_ailut import (ailut_cbackward, ailut_cforward, lut_cbackward,
                               lut_cforward)


class LUTTransformFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:

        img = img.contiguous()
        lut = lut.contiguous()

        assert img.ndimension() == 4, \
            'only support 2D image with batch and channel dimensions (4D tensor)'
        assert lut.ndimension() in [5], \
            'only support 3D lookup table with batch dimension (5D tensor)'

        output = img.new_zeros(
            (img.size(0), lut.size(1), img.size(2), img.size(3)))
        lut_cforward(img, lut, output)

        ctx.save_for_backward(img, lut)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        grad_output = grad_output.contiguous()

        img, lut = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut = torch.zeros_like(lut)

        lut_cbackward(grad_output, img, lut, grad_img, grad_lut)

        return grad_img, grad_lut


class AiLUTTransformFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, img: torch.Tensor, lut: torch.Tensor,
                vertices: torch.tensor) -> torch.Tensor:

        img = img.contiguous()
        lut = lut.contiguous()
        vertices = vertices.contiguous()

        assert img.ndimension() == 4, \
            'only support 2D image with batch and channel dimensions (4D tensor)'
        assert lut.ndimension() in [5], \
            'only support 3D lookup table with batch dimension (5D tensor)'
        assert vertices.ndimension() == 3, \
            'only support 1D vertices list with batch and channel dimensions (3D tensor)'

        output = img.new_zeros(
            (img.size(0), lut.size(1), img.size(2), img.size(3)))
        ailut_cforward(img, lut, vertices, output)

        ctx.save_for_backward(img, lut, vertices)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.contiguous()

        img, lut, vertices = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut = torch.zeros_like(lut)
        grad_ver = torch.zeros_like(vertices)

        ailut_cbackward(grad_output, img, lut, vertices, grad_img, grad_lut,
                        grad_ver)

        return grad_img, grad_lut, grad_ver


def ailut_transform(img: torch.Tensor, lut: torch.Tensor,
                    vertices: torch.Tensor) -> torch.Tensor:
    r"""Adaptive Interval 3D Lookup Table Transform (AiLUT-Transform).

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
        vertices (torch.Tensor): sampling coordinates along each dimension of
            the 3D LUT, shape (b, 3, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return AiLUTTransformFunction.apply(img, lut, vertices)


def lut_transform(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    r"""Standard 3D Lookup Table Transform.

    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return LUTTransformFunction.apply(img, lut)
