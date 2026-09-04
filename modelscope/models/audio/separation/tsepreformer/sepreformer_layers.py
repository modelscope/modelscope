# Copyright (c) Alibaba, Inc. and its affiliates.
"""T-SepReformer separator.

A fully time-domain U-Net separator: a convolutional waveform encoder, four
contracting and expanding stages built from local conv blocks plus focused
linear attention, and a convolutional decoder. Module attribute names match the
training-time model so checkpoints load without any key renaming.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..linear_attn import (GCFN, LayerScale, Linear_Attn_Template,
                           get_norm_type_conf)


class CLA(nn.Module):
    """Convolutional local attention block."""

    def __init__(self,
                 in_channels,
                 kernel_size,
                 dropout_rate,
                 Layer_scale_init=1.0e-5,
                 norm_type='layernorm'):
        super().__init__()
        norm_layer_name, norm_layer_conf = get_norm_type_conf(
            norm_type, in_channels)

        self.layer_norm = norm_layer_name(**norm_layer_conf)
        self.linear1 = nn.Linear(in_channels, in_channels * 2)
        self.GLU = nn.GLU()
        self.dw_conv_1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding='same',
            groups=in_channels)
        self.linear2 = nn.Linear(in_channels, 2 * in_channels)
        self.BN = nn.BatchNorm1d(2 * in_channels)
        self.linear3 = nn.Sequential(nn.GELU(),
                                     nn.Linear(2 * in_channels, in_channels),
                                     nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(
            dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        """(B, T, C) -> (B, T, C)."""
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.GLU(y)

        y = y.permute([0, 2, 1])
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1)

        y = self.linear2(y)
        y = y.permute(0, 2, 1)
        y = self.BN(y)
        y = y.permute(0, 2, 1)
        y = self.linear3(y)

        return x + self.Layer_scale(y)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with an optional relative position term."""

    def __init__(self,
                 n_head,
                 in_channels,
                 dropout_rate,
                 Layer_scale_init=1.0e-5,
                 norm_type='layernorm'):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head
        self.h = n_head

        norm_layer_name, norm_layer_conf = get_norm_type_conf(
            norm_type, in_channels)

        self.layer_norm = norm_layer_name(**norm_layer_conf)
        self.linear_q = nn.Linear(in_channels, in_channels)
        self.linear_k = nn.Linear(in_channels, in_channels)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.linear_out = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(
            dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x, pos_k):
        """(B, T, C) -> (B, T, C)."""
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        A = torch.matmul(q, k.transpose(-2, -1))
        if pos_k is not None:
            reshape_q = q.reshape(n_batch * self.h, -1,
                                  self.d_k).transpose(0, 1)
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0),
                                       pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
        return self.Layer_scale(self.dropout(self.linear_out(x)))


class GlobalBlock(nn.Module):
    """Linear-attention block followed by a gated conv FFN."""

    def __init__(self,
                 in_channels,
                 num_mha_heads,
                 dropout_rate,
                 ropeEmb=None,
                 ega_type='Linear_Attn_Template',
                 addition_conf={},
                 norm_type='layernorm',
                 ffn_type='GCFN'):
        super().__init__()
        assert ega_type == 'Linear_Attn_Template', ega_type
        assert ffn_type == 'GCFN', ffn_type

        ega_conf = dict(
            query_dim=in_channels,
            out_dim=in_channels,
            dim_head=in_channels // num_mha_heads,
            ropeEmb=ropeEmb)
        ega_conf.update(addition_conf.get('ega_addition_conf', {}))

        self.block = nn.ModuleDict({
            'ega':
            Linear_Attn_Template(**ega_conf),
            'gcfn':
            GCFN(
                in_channels=in_channels,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                **addition_conf.get('ffn_addition_conf', {}))
        })

    def forward(self, x, pos_k):
        """(B, C, T) -> (B, C, T)."""
        x = self.block['ega'](x, pos_k)
        x = self.block['gcfn'](x)
        return x.permute([0, 2, 1])


class LocalBlock(nn.Module):
    """Local convolutional block followed by a gated conv FFN."""

    def __init__(self,
                 in_channels,
                 kernel_size,
                 dropout_rate,
                 norm_type='layernorm',
                 ffn_type='GCFN',
                 addition_conf={},
                 cla_type='cla'):
        super().__init__()
        assert ffn_type == 'GCFN', ffn_type
        assert cla_type == 'cla', cla_type

        cla_conf = dict(
            in_channels=in_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            norm_type=norm_type)
        cla_conf.update(addition_conf.get('cla_addition_conf', {}))

        self.block = nn.ModuleDict({
            'cla':
            CLA(**cla_conf),
            'gcfn':
            GCFN(
                in_channels,
                dropout_rate,
                norm_type=norm_type,
                **addition_conf.get('ffn_addition_conf', {}))
        })

    def forward(self, x):
        """(B, T, C) -> (B, T, C)."""
        x = self.block['cla'](x)
        return self.block['gcfn'](x)


class SpkAttention(nn.Module):
    """Attention across the speaker axis."""

    def __init__(self,
                 in_channels,
                 num_mha_heads,
                 dropout_rate,
                 norm_type='layernorm',
                 ffn_type='GCFN',
                 addition_conf={}):
        super().__init__()
        assert ffn_type == 'GCFN', ffn_type

        self.self_attn = MultiHeadAttention(
            n_head=num_mha_heads,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            norm_type=norm_type)
        self.feed_forward = GCFN(
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            **addition_conf.get('ffn_addition_conf', {}))

    def forward(self, x, num_spk):
        """(B*num_spk, C, T) -> (B*num_spk, C, T)."""
        B, F_, T = x.shape
        x = x.reshape(B // num_spk, num_spk, F_, T)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape(-1, num_spk, F_)
        x = x + self.self_attn(x, None)
        x = x.reshape(B // num_spk, T, num_spk, F_)
        x = x.permute([0, 2, 3, 1])
        x = x.reshape(B, F_, T)
        x = x.permute([0, 2, 1])
        x = self.feed_forward(x)
        return x.permute([0, 2, 1])


class Masking(nn.Module):
    """Multiplicative gating against a skip tensor."""

    def __init__(self, input_dim, Activation_mask='ReLU'):
        super().__init__()
        if Activation_mask == 'Sigmoid':
            self.gate_act = nn.Sigmoid()
        elif Activation_mask == 'ReLU':
            self.gate_act = nn.ReLU()
        else:
            raise ValueError(f'Unsupported Activation_mask: {Activation_mask}')

    def forward(self, x, skip):
        return self.gate_act(x) * skip


class AudioEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 bias):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x):
        """[T] or [B, T] -> [B, C, T']."""
        x = torch.unsqueeze(
            x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(
                x, dim=1)
        x = self.conv1d(x)
        return self.gelu(x)


class FeatureProjector(nn.Module):

    def __init__(self, num_channels, in_channels, out_channels, kernel_size,
                 bias):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias)

    def forward(self, x):
        return self.conv1d(self.norm(x))


class RelativePositionalEncoding(nn.Module):

    def __init__(self, in_channels, num_heads, maxlen, embed_v=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.embedding_dim = in_channels // num_heads
        self.maxlen = maxlen
        self.pe_k = nn.Embedding(
            num_embeddings=2 * maxlen, embedding_dim=self.embedding_dim)
        self.pe_v = nn.Embedding(
            num_embeddings=2
            * maxlen, embedding_dim=self.embedding_dim) if embed_v else None

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq += self.maxlen
        pe_k_output = self.pe_k(pos_seq)
        pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
        return pe_k_output, pe_v_output


class DownConvLayer(nn.Module):

    def __init__(self, in_channels, samp_kernel_size):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=samp_kernel_size,
            stride=2,
            padding=(samp_kernel_size - 1) // 2,
            groups=in_channels)
        self.BN = nn.BatchNorm1d(num_features=in_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.permute([0, 2, 1])
        x = self.down_conv(x)
        x = self.BN(x)
        x = self.gelu(x)
        return x.permute([0, 2, 1])


class SepEncStage(nn.Module):
    """Two global/local block pairs followed by an optional 2x downsample."""

    def __init__(self,
                 global_blocks,
                 local_blocks,
                 down_conv_layer,
                 down_conv=True,
                 ropeEmb=None,
                 addition_conf={}):
        super().__init__()
        self.g_block_1 = GlobalBlock(
            **global_blocks, ropeEmb=ropeEmb, addition_conf=addition_conf)
        self.l_block_1 = LocalBlock(
            **local_blocks, addition_conf=addition_conf)

        self.g_block_2 = GlobalBlock(
            **global_blocks, ropeEmb=ropeEmb, addition_conf=addition_conf)
        self.l_block_2 = LocalBlock(
            **local_blocks, addition_conf=addition_conf)

        self.downconv = DownConvLayer(**down_conv_layer) if down_conv else None

    def forward(self, x, pos_k):
        """(B, C, T) -> downsampled (B, C, T/2) plus the pre-downsample skip."""
        x = self.g_block_1(x, pos_k)
        x = x.permute(0, 2, 1)
        x = self.l_block_1(x)
        x = x.permute(0, 2, 1)

        x = self.g_block_2(x, pos_k)
        x = x.permute(0, 2, 1)
        x = self.l_block_2(x)
        x = x.permute(0, 2, 1)

        skip = x
        if self.downconv:
            x = x.permute(0, 2, 1)
            x = self.downconv(x)
            x = x.permute(0, 2, 1)
        return x, skip


class SpkSplitStage(nn.Module):
    """Expands the channel axis into the speaker axis."""

    def __init__(self, in_channels, num_spks):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels, 4 * in_channels * num_spks, kernel_size=1),
            nn.GLU(dim=-2),
            nn.Conv1d(
                2 * in_channels * num_spks,
                in_channels * num_spks,
                kernel_size=1))
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.num_spks = num_spks

    def forward(self, x):
        """(B, C, T) -> (B*num_spks, C, T)."""
        x = self.linear(x)
        B, _, T = x.shape
        x = x.view(B * self.num_spks, -1, T)
        return self.norm(x)


class SepDecStage(nn.Module):
    """Three global/local/speaker-attention triplets."""

    def __init__(self,
                 num_spks,
                 global_blocks,
                 local_blocks,
                 spk_attention,
                 ropeEmb=None,
                 addition_conf={}):
        super().__init__()

        for i in (1, 2, 3):
            setattr(
                self, f'g_block_{i}',
                GlobalBlock(
                    **global_blocks,
                    ropeEmb=ropeEmb,
                    addition_conf=addition_conf))
            setattr(self, f'l_block_{i}',
                    LocalBlock(**local_blocks, addition_conf=addition_conf))
            setattr(self, f'spk_attn_{i}',
                    SpkAttention(**spk_attention, addition_conf=addition_conf))

        self.num_spk = num_spks

    def forward(self, x, pos_k):
        """(B*num_spk, C, T) -> (B*num_spk, C, T), returned twice."""
        for i in (1, 2, 3):
            x = getattr(self, f'g_block_{i}')(x, pos_k)
            x = x.permute(0, 2, 1)
            x = getattr(self, f'l_block_{i}')(x)
            x = x.permute(0, 2, 1)
            x = getattr(self, f'spk_attn_{i}')(x, self.num_spk)
        return x, x


class Separator(nn.Module):
    """U-Net style separator over the projected features."""

    def __init__(self,
                 num_stages,
                 relative_positional_encoding,
                 enc_stage,
                 spk_split_stage,
                 simple_fusion,
                 dec_stage,
                 first_layer_static=False,
                 linear_layers_flag=(1, 1, 1, 1),
                 use_rope_pos=False,
                 use_share_spk_split=True):
        super().__init__()

        # This port only covers the configuration of the released model.
        assert not first_layer_static, 'only first_layer_static=False is supported'
        assert not use_rope_pos, 'only use_rope_pos=False is supported'
        assert use_share_spk_split, 'only use_share_spk_split=True is supported'
        assert all(flag == 1 for flag in linear_layers_flag), (
            'only linear_layers_flag of all ones is supported')

        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(
            **relative_positional_encoding)

        self.enc_stages = nn.ModuleList([
            SepEncStage(**enc_stage, down_conv=True) for _ in range(num_stages)
        ])
        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)

        self.simple_fusion = nn.ModuleList([
            nn.Conv1d(
                in_channels=simple_fusion['out_channels'] * 2,
                out_channels=simple_fusion['out_channels'],
                kernel_size=1) for _ in range(num_stages)
        ])
        self.dec_stages = nn.ModuleList(
            [SepDecStage(**dec_stage) for _ in range(num_stages)])

    def pad_signal(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)
        elif input.dim() == 2:
            input = input.unsqueeze(1)
        elif input.dim() != 3:
            raise RuntimeError('Input can only be 2 or 3 dimensional.')
        L = 2**self.num_stages
        if torch.onnx.is_in_onnx_export():
            nframe = torch._shape_as_tensor(input)[2]
            rest = torch.remainder(-nframe, L)
            return F.pad(input, (0, rest)), rest
        nframe = input.size(2)
        rest = 0 if nframe % L == 0 else (nframe // L + 1) * L - nframe
        if rest > 0:
            pad = input.new_zeros(input.size(0), input.size(1), rest)
            input = torch.cat([input, pad], dim=-1)
        return input, rest

    def forward(self, input):
        """(B, C, L) -> last stage output plus the per-stage outputs."""
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]

        pos_seq = torch.arange(0,
                               len_x // 2**self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)

        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip.append(self.spk_split_block(skip_))
        x, _ = self.bottleneck_G(x, pos_k)
        x = self.spk_split_block(x)

        each_stage_outputs = []
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = F.interpolate(x, size=skip[idx_en].shape[-1], mode='nearest')
            x = torch.cat([x, skip[idx_en]], dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x, pos_k)

        return x, each_stage_outputs


class OutputLayer(nn.Module):
    """Projects the separator output back to the encoder feature dimension."""

    def __init__(self, in_channels, out_channels, num_spks, masking=False):
        super().__init__()
        self.masking = masking
        self.spe_block = Masking(in_channels, Activation_mask='ReLU')
        self.num_spks = num_spks
        self.end_conv1x1 = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels), nn.GLU(),
            nn.Linear(2 * out_channels, in_channels))

    def forward(self, x, input):
        """(B*num_spks, C, T) -> (num_spks, B, C_enc, T_enc)."""
        x = x[..., :input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks

        if self.masking:
            input = input.expand(self.num_spks, B, N, L).transpose(0, 1)
            input = input.reshape(B * self.num_spks, N, L)
            x = self.spe_block(x, input)

        x = x.view(B, self.num_spks, N, L)
        return x.transpose(0, 1)


class AudioDecoder(nn.ConvTranspose1d):

    def forward(self, x):
        """[B, C, L] -> [B, T]."""
        if x.dim() not in [2, 3]:
            raise RuntimeError('AudioDecoder accepts a 2/3D tensor as input')
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        return torch.squeeze(
            x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)


class TSepReformerSeparator(nn.Module):
    """Time-domain SepReformer separator.

    Args:
        num_stages: number of contracting/expanding stages.
        num_spks: number of output sources.
        module_audio_enc / module_feature_projector / module_separator /
        module_output_layer / module_audio_dec: sub-module configs.
        aux_part: build the per-stage auxiliary-loss heads. They carry weights
            in the released checkpoint, so they must be built to load it, but
            they are not evaluated during inference.
    """

    def __init__(self,
                 num_stages=4,
                 num_spks=2,
                 module_audio_enc={},
                 module_feature_projector={},
                 module_separator={},
                 module_output_layer={},
                 module_audio_dec={},
                 aux_part=True):
        super().__init__()
        self.num_stages = num_stages
        self.num_spks = num_spks
        self._num_spk = num_spks

        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)

        self.aux_part = aux_part
        self.out_layer_bn = nn.ModuleList([])
        self.decoder_bn = nn.ModuleList([])
        if aux_part:
            for _ in range(num_stages):
                self.out_layer_bn.append(
                    OutputLayer(**module_output_layer, masking=True))
                self.decoder_bn.append(AudioDecoder(**module_audio_dec))

    @property
    def num_spk(self):
        return self._num_spk

    def forward(self, input):
        """Separate a mixture waveform.

        Args:
            input (torch.Tensor): mixture waveform, [B, L].

        Returns:
            list of num_spks waveforms, each [B, L].
        """
        n_samples = input.shape[-1]

        encoder_output = self.audio_encoder(input)
        projected_feature = self.feature_projector(encoder_output)

        last_stage_output, _ = self.separator(projected_feature)

        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        audio = [
            self.audio_decoder(out_layer_output[idx])
            for idx in range(self.num_spks)
        ]
        return [pad_or_truncate(wav, n_samples) for wav in audio]


def pad_or_truncate(tensor, max_length):
    """Force a (B, T) tensor to length ``max_length``."""
    length = tensor.shape[-1]
    if length > max_length:
        return tensor[:, :max_length]
    if length < max_length:
        return F.pad(tensor, (0, max_length - length), 'constant', 0)
    return tensor
