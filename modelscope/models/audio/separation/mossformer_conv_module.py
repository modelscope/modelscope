# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Args:
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        """Returns the normalized tensor.

        Args:
            x: Tensor of size [N, C, K, S] or [N, C, L].
        """
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps)
                     + self.bias)  # yapf: disable
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps)
                     + self.bias)  # yapf: disable
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

    Args:
       dim: Dimension that you want to normalize.
       elementwise_affine: Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Args:
            x: Tensor size [N, C, K, S] or [N, C, L]
        """
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, 'out_channels should be constant multiple of in_channels'
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class MossFormerConvModule(nn.Module):
    """Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).

    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 17
        dropout_p (float, optional): probability of dropout
    """

    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 17,
                 expansion_factor: int = 2) -> None:
        super(MossFormerConvModule, self).__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, 'Currently, Only Supports expansion_factor 2'

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs (batch, time, dim): Tensor contains input sequences

        Returns:
            outputs (batch, time, dim): Tensor produces by conformer convolution module.
        """
        return inputs + self.sequential(inputs).transpose(1, 2)
