# Part of the implementation is borrowed and modified from HuggingFace XGLM,
# publicly avaialbe at https://github.com/huggingface/transformers.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['XGLM']


def sinusoidal_embedding(seq_len, dim, pad_token=None):
    half = dim // 2
    sinusoid = torch.outer(
        torch.arange(seq_len, dtype=torch.float32),
        torch.pow(10000,
                  -torch.arange(half, dtype=torch.float32).div(half - 1)))
    x = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1)
    if dim % 2 == 1:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    if pad_token is not None:
        x[pad_token, :] = 0
    return x


class SinusoidalEmbedding(nn.Module):

    def __init__(self, seq_len, dim, pad_token):
        super(SinusoidalEmbedding, self).__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.pad_token = pad_token
        self.register_buffer('weight',
                             sinusoidal_embedding(seq_len + 2, dim, pad_token))

    def forward(self, tokens):
        mask = tokens.ne(self.pad_token).long()
        indices = torch.cumsum(mask, dim=1) * mask + self.pad_token
        pos_embeds = self.weight.index_select(0, indices.view(-1)).view(
            *tokens.shape, -1)
        return pos_embeds


class GELU(nn.Module):

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1):
        assert dim % num_heads == 0
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        r"""x:      [B, L, C].
            mask:   [B, *, L, L] or None.
        """
        b, l, n, c = *x.shape[:2], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, l, n, c)
        k = self.k(x).view(b, l, n, c)
        v = self.v(x).view(b, l, n, c)

        # compute attention
        attn = self.scale * torch.einsum('binc,bjnc->bnij', q, k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # gather context
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        x = x.reshape(b, l, -1)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, ffn_act, num_heads, dropout=0.1):
        assert ffn_act in ['gelu', 'relu']
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.ffn_act = ffn_act
        self.num_heads = num_heads

        # layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            GELU() if ffn_act == 'gelu' else nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class XGLM(nn.Module):
    r"""A multilingual GPT model with an embedding head.
    """

    def __init__(self,
                 vocab_size=256008,
                 max_seq_len=2048,
                 dim=1024,
                 ffn_dim=4096,
                 ffn_act='gelu',
                 embed_dim=768,
                 num_heads=16,
                 num_layers=24,
                 pad_token=1,
                 dropout=0.1):
        super(XGLM, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.ffn_act = ffn_act
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token = pad_token
        self.scale = math.sqrt(dim)  # rescale token embedings

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim, pad_token)
        self.pos_embedding = SinusoidalEmbedding(max_seq_len, dim, pad_token)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, ffn_dim, ffn_act, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, embed_dim, bias=False)

        # causal attention mask
        self.register_buffer(
            'attn_mask',
            torch.tril(torch.ones(1, 1, 1 + max_seq_len, 1 + max_seq_len)))

        # init weights
        self.apply(self.init_weights)

    def forward(self, tokens, mask=None):
        r"""tokens: [B, L].
            mask:   [B, L].
        """
        b, seq_len = tokens.size(0), 1 + tokens.size(1)

        # embeddings
        x = self.scale * self.token_embedding(tokens)
        x = torch.cat([x, self.eos_embedding.repeat(b, 1, 1)], dim=1)
        # x = x + self.pos_embedding(tokens)
        x = self.dropout(x)

        # attention mask
        if mask is None:
            mask = self.attn_mask[:, :, :seq_len, :seq_len].repeat(b, 1, 1, 1)
        else:
            mask = self.attn_mask[:, :, :seq_len, :seq_len] * torch.cat(
                [mask, torch.zeros_like(mask[:, :1])], dim=1).view(
                    b, 1, 1, seq_len)

        # transformer
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        # head
        logits = self.head(x[:, -1])
        return logits

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                nn.init.zeros_(m.weight[m.padding_idx])
