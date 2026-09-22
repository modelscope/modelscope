# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared building blocks for the FLA-TF-Locoformer and T-SepReformer separators.

Focused linear attention and its supporting norms / gated FFN, with the
unreachable attention variants removed so that no espnet or diffusers import is
required.
"""

import torch
import torch.nn as nn


class RMSGroupNorm(nn.Module):
    """Root Mean Square Group Normalization applied to each TF bin."""

    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group**(-1.0 / 2)
        output = input / (rms + self.eps)

        output = output.view(others + (-1, ))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output


def get_norm_type_conf(norm_type, embed_dim):
    """Resolve a norm spec string such as ``layernorm`` or ``RMSGroupNorm_4``."""

    if norm_type == 'layernorm':
        return nn.LayerNorm, dict(normalized_shape=embed_dim)
    elif norm_type.startswith('RMSGroupNorm'):
        groups = int(norm_type.split('_')[1])
        return RMSGroupNorm, dict(num_groups=groups, dim=embed_dim, eps=1.0e-5)
    else:
        raise ValueError(f'norm_type {norm_type} is not supported')


class LayerScale(nn.Module):

    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        self.layer_scale = nn.Parameter(
            Layer_scale_init * torch.ones((1, ) * (dims - 1) + (input_size, )),
            requires_grad=True)

    def forward(self, x):
        return x * self.layer_scale


class FocusedLinearAttention(nn.Module):
    """Focused linear attention (FLatten-Transformer) with a depthwise value path.

    Reference: https://github.com/LeapLabTHU/FLatten-Transformer
    """

    def __init__(self,
                 dim,
                 input_dim=None,
                 num_heads=8,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 focusing_factor=3,
                 kernel_size=5,
                 mode='torch',
                 ropeEmb=None):
        super().__init__()
        assert dim % num_heads == 0, (
            f'dim {dim} should be divided by num_heads {num_heads}.')

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        if input_dim is None:
            input_dim = dim
        self.input_dim = input_dim

        self.q = nn.Linear(input_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(input_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.focusing_factor = focusing_factor
        self.dwc_kernel_size = kernel_size
        if kernel_size > 0:
            self.dwc = nn.Conv1d(
                in_channels=head_dim,
                out_channels=head_dim,
                kernel_size=kernel_size,
                groups=head_dim,
                padding=kernel_size // 2)
        else:
            self.dwc = None
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

        if mode != 'torch':
            raise ValueError(
                f'Only the torch backend is available, got mode={mode}')
        self.mode = mode
        self.ropeEmb = ropeEmb

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x)
        C = q.size(-1)

        kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        if self.ropeEmb is not None:
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                            3).contiguous()
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                            3).contiguous()
            q = self.ropeEmb.rotate_queries_or_keys(q)
            k = self.ropeEmb.rotate_queries_or_keys(k)
            q = q.permute(0, 2, 1, 3).reshape(B, N, -1)
            k = k.permute(0, 2, 1, 3).reshape(B, N, -1)

        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q**focusing_factor
        k = k**focusing_factor
        q = (q / (q.norm(dim=-1, keepdim=True) + 1e-8)) * q_norm
        k = (k / (k.norm(dim=-1, keepdim=True) + 1e-8)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                        3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1,
                                                        3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1,
                                                        3).contiguous()

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C)

        if self.dwc is not None:
            v = v.reshape(B * self.num_heads, n, -1).permute(0, 2, 1)
            x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Linear_Attn_Template(nn.Module):
    """Pre-norm linear-attention block with an optional gated residual."""

    def __init__(self,
                 linear_attn_type,
                 norm_type='layernorm',
                 mode='torch',
                 gated=False,
                 **kwargs):
        super().__init__()

        if linear_attn_type != 'FocusedLinearAttention':
            raise ValueError(
                f'Unsupported linear_attn_type: {linear_attn_type}')

        input_dim = kwargs.get('input_dim', kwargs['query_dim'])
        self.attn = FocusedLinearAttention(
            dim=kwargs['query_dim'],
            input_dim=input_dim,
            num_heads=kwargs['query_dim'] // kwargs['dim_head'],
            ropeEmb=kwargs['ropeEmb'],
            kernel_size=kwargs.get('kernel_size', 5),
            focusing_factor=kwargs.get('focusing_factor', 3),
            mode=mode)

        norm_layer_name, norm_layer_conf = get_norm_type_conf(
            norm_type, input_dim)
        self.layer_norm = norm_layer_name(**norm_layer_conf)

        self.gated = gated
        if gated:
            self.gate_layer = nn.Sequential(
                norm_layer_name(**norm_layer_conf),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid(),
            )

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # (B, C, T) -> (B, T, C)
        x = hidden_states.permute(0, 2, 1)

        _input = x
        x = self.layer_norm(x)
        x = self.attn(x)

        if self.gated:
            x = _input + self.gate_layer(_input) * x
        else:
            x = x + _input

        return x


class GCFN(nn.Module):
    """Gated convolutional feed-forward network."""

    def __init__(self,
                 in_channels,
                 dropout_rate,
                 Layer_scale_init=1.0e-5,
                 norm_type='layernorm'):
        super().__init__()

        norm_layer_name, norm_layer_conf = get_norm_type_conf(
            norm_type, in_channels)

        self.net1 = nn.Sequential(
            norm_layer_name(**norm_layer_conf),
            nn.Linear(in_channels, in_channels * 6))
        self.depthwise = nn.Conv1d(
            in_channels * 6,
            in_channels * 6,
            3,
            padding=1,
            groups=in_channels * 6)
        self.net2 = nn.Sequential(nn.GLU(), nn.Dropout(dropout_rate),
                                  nn.Linear(in_channels * 3, in_channels),
                                  nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(
            dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        """(B, T, C) -> (B, T, C)."""
        y = self.net1(x)
        y = y.permute(0, 2, 1)
        y = self.depthwise(y)
        y = y.permute(0, 2, 1)
        y = self.net2(y)
        return x + self.Layer_scale(y)
