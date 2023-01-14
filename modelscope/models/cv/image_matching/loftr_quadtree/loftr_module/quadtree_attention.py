# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from modelscope.ops.quadtree_attention import QTAttA, QTAttB


class QuadtreeAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        topks,
        value_branch=False,
        act=nn.GELU(),
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        scale=1,
        attn_type='B',
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        if attn_type == 'A':
            self.py_att = QTAttA(
                num_heads, dim // num_heads, scale=scale, topks=topks)
        else:
            self.py_att = QTAttB(
                num_heads, dim // num_heads, scale=scale, topks=topks)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = scale

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            trunc_normal_(m.weight, std=0.02)
            m.init = True
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W, msg=None):

        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        target = target.permute(0, 2, 1).reshape(B, C, H, W)
        keys = []
        values = []
        queries = []

        q = self.q_proj(x)
        k = self.k_proj(target)
        v = self.v_proj(target)
        for i in range(self.scale):
            keys.append(k)
            values.append(v)
            queries.append(q)

            if i != self.scale - 1:
                k = F.avg_pool2d(k, kernel_size=2, stride=2)
                q = F.avg_pool2d(q, kernel_size=2, stride=2)
                v = F.avg_pool2d(v, kernel_size=2, stride=2)

        msg = self.py_att(queries, keys, values).view(B, -1, C)

        x = self.proj(msg)
        x = self.proj_drop(x)

        return x
