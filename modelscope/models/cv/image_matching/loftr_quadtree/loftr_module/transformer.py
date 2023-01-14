# Part of the implementation is borrowed and modified from LoFTR,
# made public available under the Apache License, Version 2.0,
# at https://github.com/zju3dv/LoFTR

import copy
import math

import torch
import torch.nn as nn
from einops.einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .linear_attention import FullAttention, LinearAttention
from .quadtree_attention import QuadtreeAttention


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dwconv(x, H, W)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LoFTREncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention(
        ) if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead,
                                        self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead,
                                    self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask,
            kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1,
                                          self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class QuadtreeBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 topks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 scale=1,
                 attn_type='B'):

        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = QuadtreeAttention(
            dim,
            num_heads=num_heads,
            topks=topks,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale=scale,
            attn_type=attn_type)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'init'):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W):

        x = x + self.drop_path(
            self.attn(self.norm1(x), self.norm1(target), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        self.block_type = config['block_type']
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']

        if config['block_type'] == 'loftr':
            encoder_layer = LoFTREncoderLayer(config['d_model'],
                                              config['nhead'],
                                              config['attention'])
            self.layers = nn.ModuleList([
                copy.deepcopy(encoder_layer)
                for _ in range(len(self.layer_names))
            ])
            self._reset_parameters()
        elif config['block_type'] == 'quadtree':
            encoder_layer = QuadtreeBlock(
                config['d_model'],
                config['nhead'],
                attn_type=config['attn_type'],
                topks=config['topks'],
                scale=3)
            self.layers = nn.ModuleList([
                copy.deepcopy(encoder_layer)
                for _ in range(len(self.layer_names))
            ])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        if len(feat0.shape) == 4:
            B, C, H, W = feat0.shape
            feat0 = rearrange(feat0, 'b c h w -> b (h w) c')
            feat1 = rearrange(feat1, 'b c h w -> b (h w) c')

        if self.block_type == 'loftr':
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                elif name == 'cross':
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)
                else:
                    raise KeyError
        else:
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, H, W)
                    feat1 = layer(feat1, feat1, H, W)
                elif name == 'cross':
                    if self.config['block_type'] == 'quadtree':
                        feat0, feat1 = layer(feat0, feat1, H,
                                             W), layer(feat1, feat0, H, W)
                    else:
                        feat0 = layer(feat0, feat1, H, W)
                        feat1 = layer(feat1, feat0, H, W)
                else:
                    raise KeyError

        return feat0, feat1
