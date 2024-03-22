# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum, nn

from .conv_module import ConvModule, FFConvMDilated
from .fsmn import UniDeepFsmn, UniDeepFsmnDilated
from .layer_norm import CLayerNorm

# functions


def identity(t, *args, **kwargs):
    return t


def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1, ) * num_dims))


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


# scalenorm


class ScaleNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# absolute positional encodings


class ScaledSinuEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, ))
        inv_freq = 1. / (10000**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


class OffsetScale(nn.Module):

    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class FFConvM(nn.Module):

    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in), nn.Linear(dim_in, dim_out), nn.SiLU(),
            ConvModule(dim_out), nn.Dropout(dropout))

    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output


class GroupLinear(nn.Module):

    def __init__(self, dim_in, dim_out, K=4):
        super().__init__()
        hidden = dim_in // 2
        self.group_conv = nn.Conv1d(
            dim_in, hidden, groups=dim_in // K, kernel_size=1)
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, dim_out)

    def forward(
        self,
        x,
    ):
        x1 = x.transpose(2, 1)
        conv_out = self.group_conv(x1)
        x2 = self.norm(conv_out.transpose(2, 1))
        x3 = self.linear(x2)
        return x3


class FFM(nn.Module):

    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in), nn.Linear(dim_in, dim_out), nn.SiLU(),
            nn.Dropout(dropout))

    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output


# FLASH
class FLASH_ShareA_FFConvM(nn.Module):

    def __init__(self,
                 *,
                 dim,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=1.,
                 causal=False,
                 dropout=0.1,
                 rotary_pos_emb=None,
                 norm_klass=nn.LayerNorm,
                 shift_tokens=True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings
        self.rotary_pos_emb = rotary_pos_emb
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections
        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """
        # prenorm
        normed_x = x

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections
        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v,
                                          u)
        out = (att_u * v) * self.gateActivate(att_v * u)

        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k))

        # padding for groups
        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.),
                (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask,
                           torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # group along sequence
        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        attn = F.relu(sim)**2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool,
                                     device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n],
                   (quad_out_v + lin_out_v, quad_out_u + lin_out_u))


class GatedFSMNDilated(nn.Module):

    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.fsmn = UniDeepFsmnDilated(in_channels, out_channels, lorder,
                                       hidden_size)

    def forward(
        self,
        x,
    ):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x


class GatedFSMNDilatedDual(nn.Module):

    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        self.to_u = FFConvMDilated(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.to_v = FFConvMDilated(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.fsmn = UniDeepFsmnDilated(in_channels, out_channels, lorder,
                                       hidden_size)

    def forward(
        self,
        x,
    ):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x


class GatedFSMNBlockDilatedDual(nn.Module):
    """1-D convolutional block."""

    def __init__(
        self,
        dim,
        inner_channels=256,
    ):
        super(GatedFSMNBlockDilatedDual, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = GatedFSMNDilatedDual(
            inner_channels,
            inner_channels,
            lorder=20,
            hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input


class GatedFSMNBlockDilated(nn.Module):
    """1-D convolutional block."""

    def __init__(
        self,
        dim,
        inner_channels=256,
        group_size=256,
        norm_type='scalenorm',
    ):
        super(GatedFSMNBlockDilated, self).__init__()

        self.group_size = group_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        # block dilated with gating
        self.gated_fsmn = GatedFSMNDilated(
            inner_channels,
            inner_channels,
            lorder=20,
            hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input


class MossformerBlockGFSMN(nn.Module):

    def __init__(self,
                 *,
                 dim,
                 depth,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=4.,
                 causal=False,
                 attn_dropout=0.1,
                 norm_type='scalenorm',
                 shift_tokens=True):
        super().__init__()
        assert norm_type in (
            'scalenorm',
            'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.fsmn = nn.ModuleList(
            [GatedFSMNBlockDilated(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens) for _ in range(depth)
        ])

    def _build_repeats(self,
                       in_channels,
                       out_channels,
                       lorder,
                       hidden_size,
                       repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x, *, mask=None):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask=mask)
            x = self.fsmn[ii](x)
            ii = ii + 1
        return x


class MossformerBlock(nn.Module):

    def __init__(self,
                 *,
                 dim,
                 depth,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=4.,
                 causal=False,
                 attn_dropout=0.1,
                 norm_type='scalenorm',
                 shift_tokens=True):
        super().__init__()
        assert norm_type in (
            'scalenorm',
            'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens) for _ in range(depth)
        ])

    def _build_repeats(self,
                       in_channels,
                       out_channels,
                       lorder,
                       hidden_size,
                       repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x, *, mask=None):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask=mask)
            ii = ii + 1
        return x
