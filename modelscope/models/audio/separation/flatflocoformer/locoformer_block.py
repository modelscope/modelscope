# Copyright (c) Alibaba, Inc. and its affiliates.
"""FLA-TF-Locoformer separator.

A TF-Locoformer backbone whose T layers use focused linear attention (FLA) and
whose F layers use standard axial multi-head attention. Module attribute names
match the training-time model so checkpoints load without any key renaming.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..linear_attn import Linear_Attn_Template, RMSGroupNorm

# pinned to the backends the model was trained with; allowing the flash kernel
# shifts results by ~5e-7 relative
_NON_FLASH_SDPA = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]


class SwiGLUConvDeconv1d(nn.Module):
    """SwiGLU feed-forward network with an overlapping conv/deconv pair."""

    def __init__(self,
                 dim,
                 dim_inner,
                 conv1d_kernel,
                 conv1d_shift,
                 dropout=0.0):
        super().__init__()

        self.conv1d = nn.Conv1d(
            dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)
        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(
            dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """(B, seq1, seq2, C) -> (B, seq1, seq2, C), mixing along seq2."""
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        seq_len = (
            math.ceil(
                (s2 + 2 * self.diff_ks - self.conv1d_kernel)
                / self.conv1d_shift) * self.conv1d_shift + self.conv1d_kernel)
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner:, :])
        x = x[..., :self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        x = x[..., self.diff_ks:self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class FFNBranchsDecoratorsV2(nn.Module):
    """Applies one FFN branch along either the frequency or the time axis."""

    def __init__(self,
                 dim,
                 dim_inner,
                 conv1d_kernel,
                 conv1d_shift,
                 dropout=0.0,
                 TF_reverse=False):
        super().__init__()

        self.branchs = nn.ModuleList([
            SwiGLUConvDeconv1d(
                dim=dim,
                dim_inner=dim_inner,
                conv1d_kernel=conv1d_kernel,
                conv1d_shift=conv1d_shift,
                dropout=dropout)
        ])
        self.TF_reverse = TF_reverse

    def forward(self, x):
        if self.TF_reverse:
            output = x.transpose(1, 2)
            output = self.branchs[0](output)
            output = output.transpose(1, 2)
        else:
            output = self.branchs[0](x)
        return output


class LocoformerMHSABranch(nn.Module):
    """Multi-head self-attention along one axis of the (T, F) grid.

    Time layers attend across all frames of a single frequency bin, frequency
    layers across all bins of a single frame; the other axis folds into batch.
    """

    def __init__(self,
                 attention_dim,
                 n_heads=8,
                 dropout=0.0,
                 rope=None,
                 is_t_layer=False):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout
        self.rope = rope
        self.is_t_layer = is_t_layer
        self.d_k = attention_dim // n_heads

    def to_heads(self, x, seq_len):
        """(..., seq_len, C) -> (B', n_heads, seq_len, d_k)."""
        return x.reshape(-1, seq_len, self.n_heads,
                         self.d_k).permute(0, 2, 1, 3)

    def forward(self, input):
        """(3, B, T, F, C) -> (B, T, F, C)."""
        B, T_, F_, C = input.shape[1:]
        seq_len = T_ if self.is_t_layer else F_

        # fold the axis that is not attended over into the batch dimension
        qkv = input.permute(0, 1, 3, 2, 4) if self.is_t_layer else input
        query = self.to_heads(qkv[0], seq_len)
        key = self.to_heads(qkv[1], seq_len)
        value = self.to_heads(qkv[2], seq_len)

        if self.rope is not None:
            with torch.autocast(device_type=query.device.type, enabled=False):
                query = self.rope.rotate_queries_or_keys(query)
                key = self.rope.rotate_queries_or_keys(key)

        with sdpa_kernel(_NON_FLASH_SDPA):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0)

        output = output.permute(0, 2, 1, 3).reshape(-1, seq_len, C)
        if self.is_t_layer:
            return output.view(B, F_, T_, C).permute(0, 2, 1, 3)
        return output.view(B, T_, F_, C)


class LocoformerMHSA(nn.Module):
    """Axial multi-head attention with a shared QKV projection."""

    def __init__(self,
                 emb_dim,
                 attention_dim,
                 n_heads=8,
                 dropout=0.0,
                 rope_freq=None,
                 rope_time=None,
                 is_t_layer=False):
        super().__init__()
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.attention_dim = attention_dim

        self.attns = nn.ModuleList([
            LocoformerMHSABranch(
                attention_dim,
                n_heads=n_heads,
                dropout=dropout,
                rope=rope_time if is_t_layer else rope_freq,
                is_t_layer=is_t_layer)
        ])
        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        """(B, T, F, C) -> (B, T, F, C)."""
        B, T, F_, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, F_, 3, self.attention_dim)
        qkv = qkv.permute(3, 0, 1, 2, 4)
        return self.aggregate_heads(self.attns[0](qkv))


class LocoformerLinearAttnMHSA(nn.Module):
    """Linear attention along a single axis, folding the other axis into batch."""

    def __init__(self,
                 emb_dim,
                 attention_dim,
                 n_heads=8,
                 dropout=0.0,
                 is_t_layer=False,
                 addition_conf={}):
        super().__init__()

        ega_conf = dict(
            input_dim=emb_dim,
            query_dim=attention_dim,
            out_dim=emb_dim,
            dim_head=attention_dim // n_heads,
            ropeEmb=None)
        ega_conf.update(addition_conf.get('ega_addition_conf', {}))

        self.ega = Linear_Attn_Template(**ega_conf)
        self.is_t_layer = is_t_layer

    def forward(self, x):
        """(B, T, F, C) -> (B, T, F, C)."""
        B, T, F_, C = x.shape

        if self.is_t_layer:
            x = x.permute(0, 2, 3, 1).reshape(B * F_, C, T)
            x = self.ega(x, None)
            x = x.reshape(B, F_, T, C).permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 1, 3, 2).reshape(B * T, C, F_)
            x = self.ega(x, None)
            x = x.reshape(B, T, F_, C)

        return x


class LocoformerBlock(nn.Module):
    """Macaron block: FFN -> attention -> FFN, all with residual connections."""

    def __init__(self,
                 rope_freq,
                 rope_time,
                 emb_dim=128,
                 norm_type='rmsgroupnorm',
                 num_groups=4,
                 n_heads=4,
                 attention_dim=128,
                 ffn_type=('swiglu_conv1d_layerchange', ) * 2,
                 ffn_hidden_dim=(384, 384),
                 conv1d_kernel=4,
                 conv1d_shift=1,
                 dropout=0.0,
                 eps=1.0e-5,
                 is_t_layer=False,
                 attn_type='LocoformerMHSA',
                 attn_addition_conf={}):
        super().__init__()

        self.is_t_layer = is_t_layer

        def make_norm():
            if norm_type == 'rmsgroupnorm':
                return RMSGroupNorm(num_groups, emb_dim, eps=eps)
            return nn.LayerNorm(emb_dim, eps=eps)

        assert len(ffn_type) == 2, 'Macaron-style model requires two FFNs'

        # the list is built reversed, so ffn_type[0] from the config ends up as
        # the pre-attention FFN (applied with idx=-1)
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type == 'swiglu_conv1d_layerchange', f_type
            self.ffn_norm.append(make_norm())
            self.ffn.append(
                FFNBranchsDecoratorsV2(
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                    TF_reverse=is_t_layer))

        if attn_type == 'LocoformerMHSA':
            self.attn_norm = make_norm()
            self.attn = LocoformerMHSA(
                emb_dim,
                attention_dim=attention_dim,
                n_heads=n_heads,
                dropout=dropout,
                rope_freq=rope_freq,
                rope_time=rope_time,
                is_t_layer=is_t_layer)
        elif attn_type == 'LocoformerLinearAttnMHSA':
            # normalization and the residual live inside Linear_Attn_Template
            self.attn_norm = None
            self.attn = LocoformerLinearAttnMHSA(
                emb_dim,
                attention_dim=attention_dim,
                n_heads=n_heads,
                dropout=dropout,
                is_t_layer=is_t_layer,
                addition_conf=attn_addition_conf)
        else:
            raise ValueError(f'Unsupported attn_type: {attn_type}')

    def forward1(self, x, idx=0):
        input_ = x
        output = self.ffn_norm[idx](x)
        output = self.ffn[idx](output)
        return output + input_

    def forward(self, x):
        """(B, seq1, seq2, C) -> (B, seq1, seq2, C)."""
        B, T, F_, C = x.shape

        output = self.forward1(x, idx=-1)

        input_ = output
        if self.attn_norm is not None:
            output = self.attn_norm(output)
        output = self.attn(output.view([B, T, F_, C]))
        # LocoformerLinearAttnMHSA already applies a gated residual internally,
        # so T layers end up with a doubled input term; kept for checkpoint
        # compatibility with the released model.
        output = output.view([B, T, F_, C]) + input_

        return self.forward1(output, idx=0)


class FLATFLocoformerBlock(nn.Module):

    def __init__(self, rope_freq, rope_time, **kwargs):
        super().__init__()
        self.block = LocoformerBlock(rope_freq, rope_time, **kwargs)

    def forward(self, input):
        """(B, C, T, F) -> (B, C, T, F)."""
        output = input.permute(0, 2, 3, 1)
        output = self.block(output)
        return output.permute(0, 3, 1, 2)


class FLATFLocoformerSeparator(nn.Module):
    """FLA-TF-Locoformer separator.

    Args:
        num_spk: number of output sources.
        n_layers: number of blocks.
        emb_dim: hidden dimension of the encoding Conv2D.
        norm_type: 'rmsgroupnorm' or 'layernorm'.
        num_groups: number of groups in RMSGroupNorm.
        n_heads: number of attention heads.
        attention_dim: total attention dimension.
        ffn_type / ffn_hidden_dim: two-element lists (Macaron-style FFN).
        conv1d_kernel / conv1d_shift: kernel and stride of the FFN conv pair.
        use_rope: whether to apply rotary embeddings in the F layers.
        layers_type: per-layer axis, a list of 'f' / 't' of length n_layers.
        attn_types: attention classes cycled by the global layer index.
        attn_addition_conf: extra config forwarded to the linear attention.
    """

    def __init__(self,
                 num_spk=2,
                 n_layers=12,
                 emb_dim=128,
                 norm_type='rmsgroupnorm',
                 num_groups=4,
                 n_heads=4,
                 attention_dim=128,
                 ffn_type=('swiglu_conv1d_layerchange', ) * 2,
                 ffn_hidden_dim=(384, 384),
                 conv1d_kernel=4,
                 conv1d_shift=1,
                 dropout=0.0,
                 eps=1.0e-5,
                 use_rope=True,
                 layers_type=(),
                 attn_types=(),
                 attn_addition_conf={}):
        super().__init__()

        assert len(layers_type) == n_layers, (len(layers_type), n_layers)
        assert len(attn_types) > 0, 'attn_types must not be empty'
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)

        self._num_spk = num_spk
        self.n_layers = n_layers

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        if use_rope:
            rope_freq = RotaryEmbedding(attention_dim // n_heads)
            rope_time = RotaryEmbedding(attention_dim // n_heads)
        else:
            rope_freq = None
            rope_time = None

        self.f_blocks = nn.ModuleList([])
        self.t_blocks = nn.ModuleList([])
        self.layers_type = list(layers_type)

        for idx, layer_type in enumerate(self.layers_type):
            assert layer_type in ('f', 't'), layer_type
            is_t_layer = layer_type == 't'
            block = FLATFLocoformerBlock(
                rope_freq,
                rope_time,
                emb_dim=emb_dim,
                norm_type=norm_type,
                num_groups=num_groups,
                n_heads=n_heads,
                attention_dim=attention_dim,
                ffn_type=ffn_type,
                ffn_hidden_dim=ffn_hidden_dim,
                conv1d_kernel=conv1d_kernel,
                conv1d_shift=conv1d_shift,
                dropout=dropout,
                eps=eps,
                is_t_layer=is_t_layer,
                attn_type=attn_types[idx % len(attn_types)],
                attn_addition_conf=attn_addition_conf)
            if is_t_layer:
                self.t_blocks.append(block)
            else:
                self.f_blocks.append(block)

        self.deconv = nn.ConvTranspose2d(
            emb_dim, num_spk * 2, ks, padding=padding)

    @property
    def num_spk(self):
        return self._num_spk

    def forward(self, input):
        """Separate a mixture spectrogram.

        Args:
            input (torch.Tensor): complex spectrogram, [B, T, F].

        Returns:
            list of num_spk complex tensors, each [B, T, F].
        """
        batch0 = input.unsqueeze(1)  # [B, 1, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        with torch.autocast(device_type=batch.device.type, enabled=False):
            batch = self.conv(batch)  # [B, C, T, F]

        f_idx = 0
        t_idx = 0
        for layer_type in self.layers_type:
            if layer_type == 'f':
                batch = self.f_blocks[f_idx](batch)
                f_idx += 1
            else:
                batch = self.t_blocks[t_idx](batch)
                t_idx += 1

        with torch.autocast(device_type=batch.device.type, enabled=False):
            batch = self.deconv(batch)  # [B, num_spk*2, T, F]

        batch = batch.view([n_batch, self._num_spk, 2, n_frames, n_freqs])
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])
        return [batch[:, src] for src in range(self._num_spk)]
