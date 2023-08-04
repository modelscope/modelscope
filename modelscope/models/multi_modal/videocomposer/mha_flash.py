# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
import random
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attention import FlashAttention


class FlashAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 batch_size=4):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        super(FlashAttentionBlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        if context_dim is not None:
            self.context_kv = nn.Linear(context_dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1)

        if self.head_dim <= 128 and (self.head_dim % 8) == 0:
            self.flash_attn = FlashAttention(
                softmax_scale=None, attention_dropout=0.0)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.15)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.15)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, context=None):
        r"""x:       [B, C, H, W].
            context: [B, L, C] or None.
        """
        identity = x
        b, c, h, w, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).view(b, n * 3, d, h * w).chunk(3, dim=1)
        if context is not None:
            ck, cv = self.context_kv(context).reshape(b, -1, n * 2,
                                                      d).permute(0, 2, 3,
                                                                 1).chunk(
                                                                     2, dim=1)
            k = torch.cat([ck, k], dim=-1)
            v = torch.cat([cv, v], dim=-1)
            cq = torch.zeros([b, n, d, 4], dtype=q.dtype, device=q.device)
            q = torch.cat([q, cq], dim=-1)

        qkv = torch.cat([q, k, v], dim=1)
        origin_dtype = qkv.dtype
        qkv = qkv.permute(0, 3, 1, 2).reshape(b, -1, 3, n,
                                              d).half().contiguous()
        out, _ = self.flash_attn(qkv)
        out.to(origin_dtype)

        if context is not None:
            out = out[:, :-4, :, :]
        out = out.permute(0, 2, 3, 1).reshape(b, c, h, w)

        # output
        x = self.proj(out)
        return x + identity


if __name__ == '__main__':
    batch_size = 8
    flash_net = FlashAttentionBlock(
        dim=1280,
        context_dim=512,
        num_heads=None,
        head_dim=64,
        batch_size=batch_size).cuda()

    x = torch.randn([batch_size, 1280, 32, 32], dtype=torch.float32).cuda()
    context = torch.randn([batch_size, 4, 512], dtype=torch.float32).cuda()
    # context = None
    flash_net.eval()

    with amp.autocast(enabled=True):
        # warm up
        for i in range(5):
            y = flash_net(x, context)
        torch.cuda.synchronize()
        s1 = time.time()
        for i in range(10):
            y = flash_net(x, context)
        torch.cuda.synchronize()
        s2 = time.time()

    print(f'Average cost time {(s2-s1)*1000/10} ms')
