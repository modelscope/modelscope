# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum

__all__ = ['UNetSD_temporal']

USE_TEMPORAL_TRANSFORMER = True


# load all keys started with prefix and replace them with new_prefix
def load_Block(state, prefix, new_prefix=None):
    if new_prefix is None:
        new_prefix = prefix

    state_dict = {}

    state = {key: value for key, value in state.items() if prefix in key}

    for key, value in state.items():
        new_key = key.replace(prefix, new_prefix)
        state_dict[new_key] = value

    return state_dict


def load_2d_pretrained_state_dict(state, cfg):

    new_state_dict = {}

    dim = cfg.unet_dim
    num_res_blocks = cfg.unet_res_blocks
    dim_mult = cfg.unet_dim_mult
    attn_scales = cfg.unet_attn_scales

    # params
    enc_dims = [dim * u for u in [1] + dim_mult]
    dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    shortcut_dims = []
    scale = 1.0

    # embeddings
    state_dict = load_Block(state, prefix='time_embedding')
    new_state_dict.update(state_dict)
    state_dict = load_Block(state, prefix='y_embedding')
    new_state_dict.update(state_dict)
    state_dict = load_Block(state, prefix='context_embedding')
    new_state_dict.update(state_dict)

    encoder_idx = 0
    # init block
    state_dict = load_Block(
        state,
        prefix=f'encoder.{encoder_idx}',
        new_prefix=f'encoder.{encoder_idx}.0')
    new_state_dict.update(state_dict)
    encoder_idx += 1

    shortcut_dims.append(dim)
    for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
        for j in range(num_res_blocks):
            # residual (+attention) blocks
            idx = 0
            idx_ = 0
            # residual (+attention) blocks
            state_dict = load_Block(
                state,
                prefix=f'encoder.{encoder_idx}.{idx}',
                new_prefix=f'encoder.{encoder_idx}.{idx_}')
            new_state_dict.update(state_dict)
            idx += 1
            idx_ = 2

            if scale in attn_scales:
                # block.append(AttentionBlock(out_dim, context_dim, num_heads, head_dim))
                state_dict = load_Block(
                    state,
                    prefix=f'encoder.{encoder_idx}.{idx}',
                    new_prefix=f'encoder.{encoder_idx}.{idx_}')
                new_state_dict.update(state_dict)
            in_dim = out_dim
            encoder_idx += 1
            shortcut_dims.append(out_dim)

            # downsample
            if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                state_dict = load_Block(
                    state,
                    prefix='encoder.{encoder_idx}',
                    new_prefix='encoder.{encoder_idx}.0')
                new_state_dict.update(state_dict)

                shortcut_dims.append(out_dim)
                scale /= 2.0
                encoder_idx += 1

    # middle
    middle_idx = 0

    state_dict = load_Block(state, prefix=f'middle.{middle_idx}')
    new_state_dict.update(state_dict)
    middle_idx += 2

    state_dict = load_Block(
        state, prefix='middle.1', new_prefix=f'middle.{middle_idx}')
    new_state_dict.update(state_dict)
    middle_idx += 1

    for _ in range(cfg.temporal_attn_times):
        middle_idx += 1

    state_dict = load_Block(
        state, prefix='middle.2', new_prefix=f'middle.{middle_idx}')
    new_state_dict.update(state_dict)
    middle_idx += 2

    decoder_idx = 0
    for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
        for j in range(num_res_blocks + 1):
            idx = 0
            idx_ = 0
            # residual (+attention) blocks
            state_dict = load_Block(
                state,
                prefix=f'decoder.{decoder_idx}.{idx}',
                new_prefix=f'decoder.{decoder_idx}.{idx_}')
            new_state_dict.update(state_dict)
            idx += 1
            idx_ += 2
            if scale in attn_scales:
                state_dict = load_Block(
                    state,
                    prefix=f'decoder.{decoder_idx}.{idx}',
                    new_prefix=f'decoder.{decoder_idx}.{idx_}')
                new_state_dict.update(state_dict)
                idx += 1
                idx_ += 1
                for _ in range(cfg.temporal_attn_times):
                    idx_ += 1

            # upsample
            if i != len(dim_mult) - 1 and j == num_res_blocks:
                state_dict = load_Block(
                    state,
                    prefix=f'decoder.{decoder_idx}.{idx}',
                    new_prefix=f'decoder.{decoder_idx}.{idx_}')
                new_state_dict.update(state_dict)
                idx += 1
                idx_ += 2

                scale *= 2.0
            decoder_idx += 1

    state_dict = load_Block(state, prefix='head')
    new_state_dict.update(state_dict)

    return new_state_dict


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000,
                             -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


class RelativePositionBias(nn.Module):

    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact) *  # noqa
            (num_buckets - max_exact)).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


_ATTN_PRECISION = os.environ.get('ATTN_PRECISION', 'fp32')


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == 'fp32':
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_cls = CrossAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward_(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(),
                          self.checkpoint)

    def forward(self, x, context=None):
        x = self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# feedforward
class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                use_image_dataset=use_image_dataset)

    def forward(self, x, emb, batch_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, batch_size)

    def _forward(self, x, emb, batch_size):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        return h


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Resample(nn.Module):

    def __init__(self, in_dim, out_dim, mode):
        assert mode in ['none', 'upsample', 'downsample']
        super(Resample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, x, reference=None):
        if self.mode == 'upsample':
            assert reference is not None
            x = F.interpolate(x, size=reference.shape[-2:], mode='nearest')
        elif self.mode == 'downsample':
            x = F.adaptive_avg_pool2d(
                x, output_size=tuple(u // 2 for u in x.shape[-2:]))
        return x


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 use_scale_shift_norm=True,
                 mode='none',
                 dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        self.mode = mode

        # layers
        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.resample = Resample(in_dim, in_dim, mode)
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim,
                      out_dim * 2 if use_scale_shift_norm else out_dim))
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(
            in_dim, out_dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, e, reference=None):
        identity = self.resample(x, reference)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x), reference))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, context_dim=None, num_heads=None, head_dim=None):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        super(AttentionBlock, self).__init__()
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

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

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

        # compute attention
        attn = torch.matmul(q.transpose(-1, -2) * self.scale, k * self.scale)
        attn = F.softmax(attn, dim=-1)

        # gather context
        x = torch.matmul(v, attn.transpose(-1, -2))
        x = x.reshape(b, c, h, w)

        # output
        x = self.proj(x)
        return x + identity


class TemporalAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 heads=4,
                 dim_head=32,
                 rotary_emb=None,
                 use_image_dataset=False,
                 use_sim_mask=False):
        super().__init__()
        # consider num_heads first, as pos_bias needs fixed num_heads
        dim_head = dim // heads
        assert heads * dim_head == dim
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self,
                x,
                pos_bias=None,
                focus_present_mask=None,
                video_mask=None):

        identity = x
        n, height, device = x.shape[2], x.shape[-2], x.device

        x = self.norm(x)
        x = rearrange(x, 'b c f h w -> b (h w) f c')

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values （v=qkv[-1]） through to the output
            values = qkv[-1]
            out = self.to_out(values)
            out = rearrange(out, 'b (h w) f c -> b c f h w', h=height)

            return out + identity

        # split out heads
        q = rearrange(qkv[0], '... n (h d) -> ... h n d', h=self.heads)
        k = rearrange(qkv[1], '... n (h d) -> ... h n d', h=self.heads)
        v = rearrange(qkv[2], '... n (h d) -> ... h n d', h=self.heads)

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if (focus_present_mask is None and video_mask is not None):
            mask = video_mask[:, None, :] * video_mask[:, :, None]
            mask = mask.unsqueeze(1).unsqueeze(1)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        elif exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n),
                                         device=device,
                                         dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.use_sim_mask:
            sim_mask = torch.tril(
                torch.ones((n, n), device=device, dtype=torch.bool),
                diagonal=0)
            sim = sim.masked_fill(~sim_mask, -torch.finfo(sim.dtype).max)

        # numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        out = self.to_out(out)

        out = rearrange(out, 'b (h w) f c -> b c f h w', h=height)

        if self.use_image_dataset:
            out = identity + 0 * out
        else:
            out = identity + out
        return out


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 only_self_att=True,
                 multiply_zero=False):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.use_adaptor = False
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
            if self.use_adaptor:
                self.adaptor_in = nn.Linear(frames, frames)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            if self.use_adaptor:
                self.adaptor_out = nn.Linear(frames, frames)
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = rearrange(x, 'b c f h w -> (b h w) c f').contiguous()
            x = self.proj_in(x)
        if self.use_linear:
            x = rearrange(
                x, '(b f) c h w -> b (h w) f c', f=self.frames).contiguous()
            x = self.proj_in(x)

        if self.only_self_att:
            x = rearrange(x, 'bhw c f -> bhw f c').contiguous()
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            x = rearrange(x, '(b hw) f c -> b hw f c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) c f -> b hw f c', b=b).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                context[i] = rearrange(
                    context[i], '(b f) l con -> b f l con',
                    f=self.frames).contiguous()
                # calculate each batch one by one
                # (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(
                        context[i][j],
                        'f l con -> (f r) l con',
                        r=(h * w) // self.frames,
                        f=self.frames).contiguous()
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) f c -> b f c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw f c -> (b hw) c f').contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, '(b h w) c f -> b c f h w', b=b, h=h, w=w).contiguous()

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TemporalAttentionMultiBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None,
        use_image_dataset=False,
        use_sim_mask=False,
        temporal_attn_times=1,
    ):
        super().__init__()
        self.att_layers = nn.ModuleList([
            TemporalAttentionBlock(dim, heads, dim_head, rotary_emb,
                                   use_image_dataset, use_sim_mask)
            for _ in range(temporal_attn_times)
        ])

    def forward(self,
                x,
                pos_bias=None,
                focus_present_mask=None,
                video_mask=None):
        for layer in self.att_layers:
            x = layer(x, pos_bias, focus_present_mask, video_mask)
        return x


class InitTemporalConvBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(InitTemporalConvBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.use_image_dataset:
            x = identity + 0 * x
        else:
            x = identity + x
        return x


class TemporalConvBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(TemporalConvBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv2[-1].weight)
        nn.init.zeros_(self.conv2[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_image_dataset:
            x = identity + 0 * x
        else:
            x = identity + x
        return x


class TemporalConvBlock_v2(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.use_image_dataset:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


class UNetSD_temporal(nn.Module):

    def __init__(
        self,
        cfg,
        in_dim=7,
        dim=512,
        y_dim=512,
        context_dim=512,
        hist_dim=156,
        concat_dim=8,
        out_dim=6,
        dim_mult=[1, 2, 3, 4],
        num_heads=None,
        head_dim=64,
        num_res_blocks=3,
        attn_scales=[1 / 2, 1 / 4, 1 / 8],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_checkpoint=False,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        misc_dropout=0.5,
        training=True,
        inpainting=True,
        video_compositions=['text', 'mask'],
        p_all_zero=0.1,
        p_all_keep=0.1,
        zero_y=None,
        black_image_feature=None,
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(UNetSD_temporal, self).__init__()
        self.zero_y = zero_y
        self.black_image_feature = black_image_feature
        self.cfg = cfg
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.concat_dim = concat_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting
        self.video_compositions = video_compositions
        self.misc_dropout = misc_dropout
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0
        if hasattr(cfg, 'adapter_transformer_layers'
                   ) and cfg.adapter_transformer_layers:
            adapter_transformer_layers = cfg.adapter_transformer_layers
        else:
            adapter_transformer_layers = 1

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        self.pre_image_condition = nn.Sequential(
            nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 1024))

        # depth embedding
        if 'depthmap' in self.video_compositions:
            self.depth_embedding = nn.Sequential(
                nn.Conv2d(1, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.depth_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        if 'motion' in self.video_compositions:
            self.motion_embedding = nn.Sequential(
                nn.Conv2d(2, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.motion_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        # canny embedding
        if 'canny' in self.video_compositions:
            self.canny_embedding = nn.Sequential(
                nn.Conv2d(1, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.canny_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        # masked-image embedding
        if 'mask' in self.video_compositions:
            self.masked_embedding = nn.Sequential(
                nn.Conv2d(4, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim
                    * 4, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2,
                          padding=1)) if inpainting else None
            self.mask_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        # sketch embedding
        if 'sketch' in self.video_compositions:
            self.sketch_embedding = nn.Sequential(
                nn.Conv2d(1, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.sketch_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        if 'single_sketch' in self.video_compositions:
            self.single_sketch_embedding = nn.Sequential(
                nn.Conv2d(1, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.single_sketch_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        if 'local_image' in self.video_compositions:
            self.local_image_embedding = nn.Sequential(
                nn.Conv2d(3, concat_dim * 4, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((128, 128)),
                nn.Conv2d(
                    concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.local_image_embedding_after = Transformer_v2(
                heads=2,
                dim=concat_dim,
                dim_head_k=concat_dim,
                dim_head_v=concat_dim,
                dropout_atte=0.05,
                mlp_dim=concat_dim,
                dropout_ffn=0.05,
                depth=adapter_transformer_layers)

        # Condition Dropout
        self.misc_dropout = DropPath(misc_dropout)

        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(
                heads=num_heads, max_distance=32)

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        if cfg.resume:
            self.pre_image = nn.Sequential()
            init_block = nn.ModuleList(
                [nn.Conv2d(self.in_dim + concat_dim, dim, 3, padding=1)])
        else:
            self.pre_image = nn.Sequential(
                nn.Conv2d(self.in_dim + concat_dim, self.in_dim, 1, padding=0))
            init_block = nn.ModuleList(
                [nn.Conv2d(self.in_dim, dim, 3, padding=1)])

        # need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset))
            else:
                init_block.append(
                    TemporalAttentionMultiBlock(
                        dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        temporal_attn_times=temporal_attn_times,
                        use_image_dataset=use_image_dataset))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResBlock(
                        in_dim,
                        embed_dim,
                        dropout,
                        out_channels=out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    #
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True))
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset))
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        # middle
        self.middle_block = nn.ModuleList([
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ),
            SpatialTransformer(
                out_dim,
                out_dim // head_dim,
                head_dim,
                depth=1,
                context_dim=self.context_dim,
                disable_self_attn=False,
                use_linear=True)
        ])

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    ))
            else:
                self.middle_block.append(
                    TemporalAttentionMultiBlock(
                        out_dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        use_image_dataset=use_image_dataset,
                        use_sim_mask=use_sim_mask,
                        temporal_attn_times=temporal_attn_times))

        self.middle_block.append(
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList([
                    ResBlock(
                        in_dim + shortcut_dims.pop(),
                        embed_dim,
                        dropout,
                        out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True))
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset))
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(
                        out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)

    def forward(
            self,
            x,
            t,
            y=None,
            depth=None,
            image=None,
            motion=None,
            local_image=None,
            single_sketch=None,
            masked=None,
            canny=None,
            sketch=None,
            histogram=None,
            fps=None,
            video_mask=None,
            focus_present_mask=None,
            prob_focus_present=0.,
            mask_last_frame_num=0  # mask last frame num
    ):

        assert self.inpainting or masked is None, 'inpainting is not supported'

        batch, c, f, h, w = x.shape
        device = x.device
        self.batch = batch

        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like(
                    (batch, ), prob_focus_present, device=device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(
                x.shape[2], device=x.device)
        else:
            time_rel_pos_bias = None

        # all-zero and all-keep masks
        zero = torch.zeros(batch, dtype=torch.bool).to(x.device)
        keep = torch.zeros(batch, dtype=torch.bool).to(x.device)
        if self.training:
            nzero = (torch.rand(batch) < self.p_all_zero).sum()
            nkeep = (torch.rand(batch) < self.p_all_keep).sum()
            index = torch.randperm(batch)
            zero[index[0:nzero]] = True
            keep[index[nzero:nzero + nkeep]] = True
        assert not (zero & keep).any()
        misc_dropout = partial(self.misc_dropout, zero=zero, keep=keep)

        concat = x.new_zeros(batch, self.concat_dim, f, h, w)
        if depth is not None:
            # DropPath mask
            depth = rearrange(depth, 'b c f h w -> (b f) c h w')
            depth = self.depth_embedding(depth)
            h = depth.shape[2]
            depth = self.depth_embedding_after(
                rearrange(depth, '(b f) c h w -> (b h w) f c', b=batch))

            #
            depth = rearrange(depth, '(b h w) f c -> b c f h w', b=batch, h=h)
            concat = concat + misc_dropout(depth)

        # local_image_embedding
        if local_image is not None:
            local_image = rearrange(local_image, 'b c f h w -> (b f) c h w')
            local_image = self.local_image_embedding(local_image)

            h = local_image.shape[2]
            local_image = self.local_image_embedding_after(
                rearrange(local_image, '(b f) c h w -> (b h w) f c', b=batch))
            local_image = rearrange(
                local_image, '(b h w) f c -> b c f h w', b=batch, h=h)

            concat = concat + misc_dropout(local_image)

        if motion is not None:
            motion = rearrange(motion, 'b c f h w -> (b f) c h w')
            motion = self.motion_embedding(motion)

            h = motion.shape[2]
            motion = self.motion_embedding_after(
                rearrange(motion, '(b f) c h w -> (b h w) f c', b=batch))
            motion = rearrange(
                motion, '(b h w) f c -> b c f h w', b=batch, h=h)

            if hasattr(self.cfg, 'p_zero_motion_alone'
                       ) and self.cfg.p_zero_motion_alone and self.training:
                motion_d = torch.rand(batch) < self.cfg.p_zero_motion
                motion_d = motion_d[:, None, None, None, None]
                motion = motion.masked_fill(motion_d.cuda(), 0)
                concat = concat + motion
            else:
                concat = concat + misc_dropout(motion)

        if canny is not None:
            # DropPath mask
            canny = rearrange(canny, 'b c f h w -> (b f) c h w')
            canny = self.canny_embedding(canny)

            h = canny.shape[2]
            canny = self.canny_embedding_after(
                rearrange(canny, '(b f) c h w -> (b h w) f c', b=batch))
            canny = rearrange(canny, '(b h w) f c -> b c f h w', b=batch, h=h)

            concat = concat + misc_dropout(canny)

        if sketch is not None:
            # DropPath mask
            sketch = rearrange(sketch, 'b c f h w -> (b f) c h w')
            sketch = self.sketch_embedding(sketch)

            h = sketch.shape[2]
            sketch = self.sketch_embedding_after(
                rearrange(sketch, '(b f) c h w -> (b h w) f c', b=batch))
            sketch = rearrange(
                sketch, '(b h w) f c -> b c f h w', b=batch, h=h)

            concat = concat + misc_dropout(sketch)

        if single_sketch is not None:
            # DropPath mask
            single_sketch = rearrange(single_sketch,
                                      'b c f h w -> (b f) c h w')
            single_sketch = self.single_sketch_embedding(single_sketch)

            h = single_sketch.shape[2]
            single_sketch = self.single_sketch_embedding_after(
                rearrange(
                    single_sketch, '(b f) c h w -> (b h w) f c', b=batch))
            single_sketch = rearrange(
                single_sketch, '(b h w) f c -> b c f h w', b=batch, h=h)

            concat = concat + misc_dropout(single_sketch)

        if masked is not None:
            # DropPath mask
            masked = rearrange(masked, 'b c f h w -> (b f) c h w')
            masked = self.masked_embedding(masked)

            h = masked.shape[2]
            masked = self.mask_embedding_after(
                rearrange(masked, '(b f) c h w -> (b h w) f c', b=batch))
            masked = rearrange(
                masked, '(b h w) f c -> b c f h w', b=batch, h=h)

            concat = concat + misc_dropout(masked)

        x = torch.cat([x, concat], dim=1)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.pre_image(x)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)

        # embeddings
        if self.use_fps_condition and fps is not None:
            e = self.time_embed(sinusoidal_embedding(
                t, self.dim)) + self.fps_embedding(
                    sinusoidal_embedding(fps, self.dim))
        else:
            e = self.time_embed(sinusoidal_embedding(t, self.dim))

        context = x.new_zeros(batch, 0, self.context_dim)
        if y is not None:
            y_context = misc_dropout(y)
            context = torch.cat([context, y_context], dim=1)
        else:
            y_context = self.zero_y.repeat(batch, 1, 1)
            context = torch.cat([context, y_context], dim=1)

        if image is not None:
            image_context = misc_dropout(self.pre_image_condition(image))
            context = torch.cat([context, image_context], dim=1)

        # repeat f times for spatial e and context
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask)

        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(
                block,
                x,
                e,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        return x

    def _forward_single(self,
                        module,
                        x,
                        e,
                        context,
                        time_rel_pos_bias,
                        focus_present_mask,
                        video_mask,
                        reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(
                module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context,
                                         time_rel_pos_bias, focus_present_mask,
                                         video_mask, reference)
        else:
            x = module(x)
        return x


class PreNormattention(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class PreNormattention_qkv(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        _, _, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention_qkv(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, q, k, v):
        _, _, _, h = *q.shape, self.heads
        bk = k.shape[0]
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PostNormattention(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)


class Transformer_v2(nn.Module):

    def __init__(self,
                 heads=8,
                 dim=2048,
                 dim_head_k=256,
                 dim_head_v=256,
                 dropout_atte=0.05,
                 mlp_dim=2048,
                 dropout_ffn=0.05,
                 depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNormattention(
                        dim,
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head_k,
                            dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, x):
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class DropPath(nn.Module):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep.
    """

    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args

        # params
        x = args[0]
        b = x.size(0)
        n = (torch.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=torch.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = torch.where(mask)[0]
        index = index[torch.randperm(len(index))[:n]]
        if zero is not None:
            index = torch.cat([index, torch.where(zero)[0]], dim=0)

        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output

    def broadcast(self, src, dst):
        assert src.size(0) == dst.size(0)
        shape = (dst.size(0), ) + (1, ) * (dst.ndim - 1)
        return src.view(shape)


if __name__ == '__main__':
    from config import cfg
    # [model] unet
    model = UNetSD_temporal(
        in_dim=cfg.unet_in_dim,
        dim=cfg.unet_dim,
        y_dim=cfg.unet_y_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attn_times=0,
        use_checkpoint=cfg.use_checkpoint,
        use_image_dataset=True,
        use_fps_condition=cfg.use_fps_condition)

    print(
        int(sum(p.numel() for k, p in model.named_parameters()) / (1024**2)),
        'M parameters')
