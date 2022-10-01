# Part of the implementation is borrowed and modified from CLIP, publicly avaialbe at https://github.com/openai/CLIP.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import modelscope.models.cv.image_to_image_translation.ops as ops  # for using differentiable all_gather

__all__ = [
    'CLIP', 'clip_vit_b_32', 'clip_vit_b_16', 'clip_vit_l_14',
    'clip_vit_l_14_336px', 'clip_vit_h_16'
]


def to_fp16(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data = m.weight.data.half()
        if m.bias is not None:
            m.bias.data = m.bias.data.half()
    elif hasattr(m, 'head'):
        p = getattr(m, 'head')
        p.data = p.data.half()


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    r"""Subclass of nn.LayerNorm to handle fp16.
    """

    def forward(self, x):
        return super(LayerNorm, self).forward(x.float()).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        assert dim % num_heads == 0
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, mask=None):
        r"""x:      [B, L, C].
            mask:   [*, L, L].
        """
        b, l, _, n = *x.size(), self.num_heads

        # compute query, key, and value
        q, k, v = self.to_qkv(x.transpose(0, 1)).chunk(3, dim=-1)
        q = q.reshape(l, b * n, -1).transpose(0, 1)
        k = k.reshape(l, b * n, -1).transpose(0, 1)
        v = v.reshape(l, b * n, -1).transpose(0, 1)

        # compute attention
        attn = self.scale * torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            attn = attn.masked_fill(mask[:, :l, :l] == 0, float('-inf'))
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        attn = self.attn_dropout(attn)

        # gather context
        x = torch.bmm(attn, v)
        x = x.view(b, n, l, -1).transpose(1, 2).reshape(b, l, -1)

        # output
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # layers
        self.norm1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, attn_dropout, proj_dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), QuickGELU(), nn.Linear(dim * 4, dim),
            nn.Dropout(proj_dropout))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0):
        assert image_size % patch_size == 0
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size)**2

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            gain * torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim)
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, num_heads, attn_dropout, proj_dropout)
            for _ in range(num_layers)
        ])
        self.post_norm = LayerNorm(dim)

        # head
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        b, dtype = x.size(0), self.head.dtype
        x = x.type(dtype)

        # patch-embedding
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)  # [b, n, c]
        x = torch.cat([self.cls_embedding.repeat(b, 1, 1).type(dtype), x],
                      dim=1)
        x = self.dropout(x + self.pos_embedding.type(dtype))
        x = self.pre_norm(x)

        # transformer
        x = self.transformer(x)

        # head
        x = self.post_norm(x)
        x = torch.mm(x[:, 0, :], self.head)
        return x

    def fp16(self):
        return self.apply(to_fp16)


class TextTransformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 text_len,
                 dim=512,
                 out_dim=512,
                 num_heads=8,
                 num_layers=12,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0):
        super(TextTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(0.01 * torch.randn(1, text_len, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.transformer = nn.ModuleList([
            AttentionBlock(dim, num_heads, attn_dropout, proj_dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(dim)

        # head
        gain = 1.0 / math.sqrt(dim)
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

        # causal attention mask
        self.register_buffer('attn_mask',
                             torch.tril(torch.ones(1, text_len, text_len)))

    def forward(self, x):
        eot, dtype = x.argmax(dim=-1), self.head.dtype

        # embeddings
        x = self.dropout(
            self.token_embedding(x).type(dtype)
            + self.pos_embedding.type(dtype))

        # transformer
        for block in self.transformer:
            x = block(x, self.attn_mask)

        # head
        x = self.norm(x)
        x = torch.mm(x[torch.arange(x.size(0)), eot], self.head)
        return x

    def fp16(self):
        return self.apply(to_fp16)


class CLIP(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 image_size=224,
                 patch_size=16,
                 vision_dim=768,
                 vision_heads=12,
                 vision_layers=12,
                 vocab_size=49408,
                 text_len=77,
                 text_dim=512,
                 text_heads=8,
                 text_layers=12,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0):
        super(CLIP, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout)
        self.textual = TextTransformer(
            vocab_size=vocab_size,
            text_len=text_len,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(self, imgs, txt_tokens):
        r"""imgs:       [B, C, H, W] of torch.float32.
            txt_tokens: [B, T] of torch.long.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_tokens)

        # normalize features
        xi = F.normalize(xi, p=2, dim=1)
        xt = F.normalize(xt, p=2, dim=1)

        # gather features from all ranks
        full_xi = ops.diff_all_gather(xi)
        full_xt = ops.diff_all_gather(xt)

        # logits
        scale = self.log_scale.exp()
        logits_i2t = scale * torch.mm(xi, full_xt.t())
        logits_t2i = scale * torch.mm(xt, full_xi.t())

        # labels
        labels = torch.arange(
            len(xi) * ops.get_rank(),
            len(xi) * (ops.get_rank() + 1),
            dtype=torch.long,
            device=xi.device)
        return logits_i2t, logits_t2i, labels

    def init_weights(self):
        # embeddings
        nn.init.normal_(self.textual.token_embedding.weight, std=0.02)
        nn.init.normal_(self.visual.patch_embedding.weight, tsd=0.1)

        # attentions
        for modality in ['visual', 'textual']:
            dim = self.vision_dim if modality == 'visual' else 'textual'
            transformer = getattr(self, modality).transformer
            proj_gain = (1.0 / math.sqrt(dim)) * (
                1.0 / math.sqrt(2 * transformer.num_layers))
            attn_gain = 1.0 / math.sqrt(dim)
            mlp_gain = 1.0 / math.sqrt(2.0 * dim)
            for block in transformer.layers:
                nn.init.normal_(block.attn.to_qkv.weight, std=attn_gain)
                nn.init.normal_(block.attn.proj.weight, std=proj_gain)
                nn.init.normal_(block.mlp[0].weight, std=mlp_gain)
                nn.init.normal_(block.mlp[2].weight, std=proj_gain)

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay':
            0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups

    def fp16(self):
        return self.apply(to_fp16)


def clip_vit_b_32(**kwargs):
    return CLIP(
        embed_dim=512,
        image_size=224,
        patch_size=32,
        vision_dim=768,
        vision_heads=12,
        vision_layers=12,
        text_dim=512,
        text_heads=8,
        text_layers=12,
        **kwargs)


def clip_vit_b_16(**kwargs):
    return CLIP(
        embed_dim=512,
        image_size=224,
        patch_size=16,
        vision_dim=768,
        vision_heads=12,
        vision_layers=12,
        text_dim=512,
        text_heads=8,
        text_layers=12,
        **kwargs)


def clip_vit_l_14(**kwargs):
    return CLIP(
        embed_dim=768,
        image_size=224,
        patch_size=14,
        vision_dim=1024,
        vision_heads=16,
        vision_layers=24,
        text_dim=768,
        text_heads=12,
        text_layers=12,
        **kwargs)


def clip_vit_l_14_336px(**kwargs):
    return CLIP(
        embed_dim=768,
        image_size=336,
        patch_size=14,
        vision_dim=1024,
        vision_heads=16,
        vision_layers=24,
        text_dim=768,
        text_heads=12,
        text_layers=12,
        **kwargs)


def clip_vit_h_16(**kwargs):
    return CLIP(
        embed_dim=1024,
        image_size=256,
        patch_size=16,
        vision_dim=1280,
        vision_heads=16,
        vision_layers=32,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        **kwargs)
