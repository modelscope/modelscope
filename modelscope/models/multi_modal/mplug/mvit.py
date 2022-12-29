# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

MViTv2_Base_config = {
    'depth':
    24,
    'dim_mul': [[2, 2.0], [5, 2.0], [21, 2.0]],
    'head_mul': [[2, 2.0], [5, 2.0], [21, 2.0]],
    'pool_q_stride':
    [[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1],
     [5, 1, 2, 2], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1],
     [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1],
     [15, 1, 1, 1], [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1],
     [20, 1, 1, 1], [21, 1, 2, 2], [22, 1, 1, 1], [23, 1, 1, 1]],
    'pool_kvq_kernel': [3, 3, 3],
    'pool_kv_stride_adaptive': [1, 4, 4],
}


def interpolate_rel_pos_embed(state_dict_origin,
                              state_dict_model,
                              temporal=True,
                              verbose=False):
    rel_pos_embed_types = ['rel_pos_h', 'rel_pos_w']
    if temporal:
        rel_pos_embed_types += ['rel_pos_t']

    state_dict_inflated = state_dict_origin.copy()
    for k, v2d in state_dict_origin.items():
        if any([x in k for x in rel_pos_embed_types]):
            v3d = state_dict_model[k]
            if v2d.shape[0] != v3d.shape[0]:
                rel_pos_resized = F.interpolate(
                    v2d.reshape(1, v2d.shape[0], -1).permute(0, 2, 1),
                    size=v3d.shape[0],
                    mode='linear',
                )
                v3d = rel_pos_resized.reshape(-1, v3d.shape[0]).permute(1, 0)
                if verbose:
                    print('Inflate {}: {} -> {}: {}'.format(
                        k, v2d.shape, k, v3d.shape))
            else:
                v3d = v2d
            state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def _prepare_mvit_configs(cfg):
    depth = cfg['depth']
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    for i in range(len(cfg['dim_mul'])):
        dim_mul[cfg['dim_mul'][i][0]] = cfg['dim_mul'][i][1]
    for i in range(len(cfg['head_mul'])):
        head_mul[cfg['head_mul'][i][0]] = cfg['head_mul'][i][1]

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    for i in range(len(cfg['pool_q_stride'])):
        stride_q[cfg['pool_q_stride'][i][0]] = cfg['pool_q_stride'][i][1:]
        pool_q[cfg['pool_q_stride'][i][0]] = cfg['pool_kvq_kernel']

    if cfg['pool_kv_stride_adaptive'] is not None:
        _stride_kv = cfg['pool_kv_stride_adaptive']
        cfg['pool_kv_stride'] = []
        for i in range(cfg['depth']):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            cfg['pool_kv_stride'].append([i] + _stride_kv)

    for i in range(len(cfg['pool_kv_stride'])):
        stride_kv[cfg['pool_kv_stride'][i][0]] = cfg['pool_kv_stride'][i][1:]
        pool_kv[cfg['pool_kv_stride'][i][0]] = cfg['pool_kvq_kernel']

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        print(f'min width {min_width}')
        print(f'width {width} divisor {divisor}')
        print(f'other {int(width + divisor / 2) // divisor * divisor}')

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
            self,
            dim_in=3,
            dim_out=768,
            kernel=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
            conv2d=False,
    ):
        super().__init__()

        if conv2d:
            conv_function = nn.Conv2d
        else:
            conv_function = nn.Conv3d

        self.proj = conv_function(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(
            f'Unsupported input dimension {tensor.shape}')

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous())

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  # tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode='linear',
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)


def cal_rel_pos_spatial(attn, q, k, has_cls_embed, q_shape, k_shape, rel_pos_h,
                        rel_pos_w):
    """
    Decomposed Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio
        - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio
        - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum('bythwc,hkc->bythwk', r_q,
                           Rh)  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum('bythwc,wkc->bythwk', r_q,
                           Rw)  # [B, H, q_t, qh, qw, k_w]

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]).view(B, -1, q_t * q_h * q_w,
                                                      k_t * k_h * k_w)

    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    """
    Temporal Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (
        torch.arange(q_t)[:, None] * q_t_ratio
        - torch.arange(k_t)[None, :] * k_t_ratio)
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w,
                                                dim)

    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]).view(B, -1, q_t * q_h * q_w,
                                                  k_t * k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):

    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode='conv',
        # If True, perform pool before projection.
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
        separate_qkv=False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.separate_qkv = separate_qkv
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first or separate_qkv:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ('avg', 'max'):
            pool_op = nn.MaxPool3d if mode == 'max' else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0 else None)
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0 else None)
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0 else None)
        elif mode == 'conv' or mode == 'conv_unshared':
            if pool_first:
                dim_conv = dim // num_heads if mode == 'conv' else dim
            else:
                dim_conv = dim_out // num_heads if mode == 'conv' else dim_out
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                ) if len(kernel_q) > 0 else None)
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                ) if len(kernel_kv) > 0 else None)
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                ) if len(kernel_kv) > 0 else None)
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f'Unsupported model {mode}')

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            size = input_size[1]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)
        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            # if not rel_pos_zero_init:
            #     trunc_normal_(self.rel_pos_t, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, thw_shape):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == 'conv_unshared':
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != 'conv_unshared'
            if not self.separate_qkv:
                qkv = (
                    self.qkv(x).reshape(B, N, 3, self.num_heads,
                                        -1).permute(2, 0, 3, 1, 4))
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                q = k = v = x
                q = (
                    self.q(q).reshape(B, N, self.num_heads,
                                      -1).permute(0, 2, 1, 3))
                k = (
                    self.k(k).reshape(B, N, self.num_heads,
                                      -1).permute(0, 2, 1, 3))
                v = (
                    self.v(v).reshape(B, N, self.num_heads,
                                      -1).permute(0, 2, 1, 3))

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, 'norm_q') else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, 'norm_k') else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, 'norm_v') else None,
        )

        if self.pool_first:
            q_N = (
                np.prod(q_shape)
                + 1 if self.has_cls_embed else np.prod(q_shape))
            k_N = (
                np.prod(k_shape)
                + 1 if self.has_cls_embed else np.prod(k_shape))
            v_N = (
                np.prod(v_shape)
                + 1 if self.has_cls_embed else np.prod(v_shape))

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = (
                self.q(q).reshape(B, q_N, self.num_heads,
                                  -1).permute(0, 2, 1, 3))

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = (
                self.v(v).reshape(B, v_N, self.num_heads,
                                  -1).permute(0, 2, 1, 3))

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = (
                self.k(k).reshape(B, k_N, self.num_heads,
                                  -1).permute(0, 2, 1, 3))

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                k,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self.residual_pooling:
            # Minor Difference
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape


class MultiScaleBlock(nn.Module):

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode='conv',
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
        dim_mul_in_att=False,
        separate_qkv=False,
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        att_dim = dim_out if dim_mul_in_att else dim

        self.use_grad_checkpoint = use_grad_checkpoint

        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            separate_qkv=separate_qkv,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0 else None)

    def forward(self, x, thw_shape):
        x_norm = self.norm1(x)
        if self.use_grad_checkpoint:
            x_block, thw_shape_new = checkpoint.checkpoint(
                self.attn, x_norm, thw_shape)
        else:
            x_block, thw_shape_new = self.attn(x_norm, thw_shape)

        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed)
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        if self.use_grad_checkpoint:
            x_mlp = checkpoint.checkpoint(self.mlp, x_norm)
        else:
            x_mlp = self.mlp(x_norm)

        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


class MViTv2(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526
    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(
        self,
        img_size=224,
        embed_dim=96,
        num_classes=1000,
        num_frames=4,
        num_heads=1,
        depth=24,
        patch_kernel=[3, 7, 7],
        patch_stride=[2, 4, 4],
        patch_padding=[1, 3, 3],
        config=None,
        dropout_rate=0.,
        drop_path_rate=0.,
        mlp_ratio=4.,
        qkv_bias=True,
        mode='conv',
        cls_embed_on=True,
        use_abs_pos=False,
        rel_pos_spatial=True,
        rel_pos_temporal=True,
        rel_pos_zero_init=False,
        residual_pooling=True,
        dim_mul_in_att=True,
        pool_first=False,
        zero_decay_pos_cls=False,
        separate_qkv=False,
        norm_stem=False,
        sep_pos_embed=False,
        use_grad_checkpoint=True,
    ):
        super().__init__()
        # Prepare input.
        in_chans = 3
        self.img_size = img_size
        # Prepare output.
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # MViT params.
        self.num_heads = num_heads
        self.depth = depth
        self.cls_embed_on = cls_embed_on
        self.use_abs_pos = use_abs_pos
        self.zero_decay_pos_cls = zero_decay_pos_cls
        self.use_grad_checkpoint = use_grad_checkpoint
        self.sep_pos_embed = sep_pos_embed
        self.drop_rate = dropout_rate

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if use_grad_checkpoint:
            self.patch_embed = checkpoint_wrapper(
                PatchEmbed(
                    dim_in=in_chans,
                    dim_out=embed_dim,
                    kernel=patch_kernel,
                    stride=patch_stride,
                    padding=patch_padding,
                ))
        else:
            self.patch_embed = PatchEmbed(
                dim_in=in_chans,
                dim_out=embed_dim,
                kernel=patch_kernel,
                stride=patch_stride,
                padding=patch_padding,
            )

        patch_dims = [
            num_frames // patch_stride[0],
            img_size // patch_stride[1],
            img_size // patch_stride[2],
        ]
        num_patches = np.prod(patch_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim))

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(1, self.patch_dims[1] * self.patch_dims[2],
                                embed_dim))
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim))
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, pos_embed_dim, embed_dim))

        assert config is not None
        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs(
            config)
        input_size = patch_dims

        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if dim_mul_in_att:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=rel_pos_spatial,
                rel_pos_temporal=rel_pos_temporal,
                rel_pos_zero_init=rel_pos_zero_init,
                residual_pooling=residual_pooling,
                dim_mul_in_att=dim_mul_in_att,
                separate_qkv=separate_qkv,
                use_grad_checkpoint=False)
            if use_grad_checkpoint:
                attention_block = checkpoint_wrapper(
                    attention_block, offload_to_cpu=False)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        self.head = nn.Identity()

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend([
                        'pos_embed_spatial',
                        'pos_embed_temporal',
                        'pos_embed_class',
                    ])
                else:
                    names.append(['pos_embed'])
            if self.rel_pos_spatial:
                names.extend(['rel_pos_h', 'rel_pos_w', 'rel_pos_hw'])
            if self.rel_pos_temporal:
                names.extend(['rel_pos_t'])
            if self.cls_embed_on:
                names.append('cls_token')

        return names

    def _get_pos_embed(self, pos_embed, bcthw):
        t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :].reshape(1, p_t, p_h, p_w,
                                           -1).permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode='trilinear',
            )
            pos_embed = new_pos_embed.reshape(1, -1,
                                              t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward_features(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x, bcthw = self.patch_embed(x)

        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1) + torch.repeat_interleave(
                        self.pos_embed_temporal,
                        self.patch_dims[1] * self.patch_dims[2],
                        dim=1)
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                pos_embed = self._get_pos_embed(pos_embed, bcthw)
                x = x + pos_embed
            else:
                pos_embed = self._get_pos_embed(self.pos_embed, bcthw)
                x = x + pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
