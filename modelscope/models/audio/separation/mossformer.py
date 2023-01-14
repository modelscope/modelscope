# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.separation.mossformer_block import (
    MossFormerModule, ScaledSinuEmbedding)
from modelscope.models.audio.separation.mossformer_conv_module import (
    CumulativeLayerNorm, GlobalLayerNorm)
from modelscope.models.base import Tensor
from modelscope.utils.constant import Tasks

EPS = 1e-8


@MODELS.register_module(
    Tasks.speech_separation,
    module_name=Models.speech_mossformer_separation_temporal_8k)
class MossFormer(TorchModel):
    """Library to support MossFormer speech separation.

        Args:
            model_dir (str): the model path.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.encoder = Encoder(
            kernel_size=kwargs['kernel_size'],
            out_channels=kwargs['out_channels'])
        self.decoder = Decoder(
            in_channels=kwargs['in_channels'],
            out_channels=1,
            kernel_size=kwargs['kernel_size'],
            stride=kwargs['stride'],
            bias=kwargs['bias'])
        self.mask_net = MossFormerMaskNet(
            kwargs['in_channels'],
            kwargs['out_channels'],
            MossFormerM(kwargs['num_blocks'], kwargs['d_model'],
                        kwargs['attn_dropout'], kwargs['group_size'],
                        kwargs['query_key_dim'], kwargs['expansion_factor'],
                        kwargs['causal']),
            norm=kwargs['norm'],
            num_spks=kwargs['num_spks'])
        self.num_spks = kwargs['num_spks']

    def forward(self, inputs: Tensor) -> Dict[str, Any]:
        # Separation
        mix_w = self.encoder(inputs)
        est_mask = self.mask_net(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask
        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )
        # T changed after conv1d in encoder, fix it here
        t_origin = inputs.size(1)
        t_est = est_source.size(1)
        if t_origin > t_est:
            est_source = F.pad(est_source, (0, 0, 0, t_origin - t_est))
        else:
            est_source = est_source[:, :t_origin, :]
        return est_source

    def load_check_point(self, load_path=None, device=None):
        if not load_path:
            load_path = self.model_dir
        if not device:
            device = torch.device('cpu')
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(load_path, 'encoder.bin'), map_location=device),
            strict=True)
        self.decoder.load_state_dict(
            torch.load(
                os.path.join(load_path, 'decoder.bin'), map_location=device),
            strict=True)
        self.mask_net.load_state_dict(
            torch.load(
                os.path.join(load_path, 'masknet.bin'), map_location=device),
            strict=True)

    def as_dict(self):
        return dict(
            encoder=self.encoder, decoder=self.decoder, masknet=self.mask_net)


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Args:
        kernel_size: Length of filters.
        in_channels: Number of  input channels.
        out_channels: Number of output channels.

    Example:
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self,
                 kernel_size: int = 2,
                 out_channels: int = 64,
                 in_channels: int = 1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        """Return the encoded output.

        Args:
            x: Input tensor with dimensionality [B, L].

        Returns:
            Encoded tensor with dimensionality [B, N, T_out].
            where B = Batchsize
                  L = Number of timepoints
                  N = Number of filters
                  T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Args:
        kernel_size: Length of filters.
        in_channels: Number of  input channels.
        out_channels: Number of output channels.

    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Args:
            x: Input tensor with dimensionality [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError('{} accept 3/4D tensor as input'.format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class IdentityBlock:
    """This block is used when we want to have identity transformation within the Dual_path block.

    Example
    -------
    >>> x = torch.randn(10, 100)
    >>> IB = IdentityBlock()
    >>> xhat = IB(x)
    """

    def _init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x


class MossFormerM(nn.Module):
    """This class implements the transformer encoder.

    Args:
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512)) #B, S, N
    >>> net = MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(self,
                 num_blocks,
                 d_model=None,
                 attn_dropout=0.1,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=4.,
                 causal=False):
        super().__init__()

        self.mossformerM = MossFormerModule(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout)
        import speechbrain as sb
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(self, src: torch.Tensor):
        """
        Args:
            src: Tensor shape [B, S, N],
            where, B = Batchsize,
                   S = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output


class ComputeAttention(nn.Module):
    """Computation block for dual-path processing.

    Args:
    att_mdl : torch.nn.module
        Model to process within the chunks.
     out_channels : int
        Dimensionality of attention model.
     norm : str
        Normalization type.
     skip_connection : bool
        Skip connection around the attention module.

    Example
    ---------
        >>> att_block = MossFormerM(num_blocks=8, d_model=512)
        >>> comp_att = ComputeAttention(att_block, 512)
        >>> x = torch.randn(10, 64, 512)
        >>> x = comp_att(x)
        >>> x.shape
        torch.Size([10, 64, 512])
    """

    def __init__(
        self,
        att_mdl,
        out_channels,
        norm='ln',
        skip_connection=True,
    ):
        super(ComputeAttention, self).__init__()

        self.att_mdl = att_mdl
        self.skip_connection = skip_connection

        # Norm
        self.norm = norm
        if norm is not None:
            self.att_norm = select_norm(norm, out_channels, 3)

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Args:
            x: Input tensor of dimension [B, S, N].

        Returns:
            out: Output tensor of dimension [B, S, N].
            where, B = Batchsize,
               N = number of filters
               S = time points
        """
        # [B, S, N]
        att_out = x.permute(0, 2, 1).contiguous()

        att_out = self.att_mdl(att_out)

        # [B, N, S]
        att_out = att_out.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            att_out = self.att_norm(att_out)

        # [B, N, S]
        if self.skip_connection:
            att_out = att_out + x

        out = att_out
        return out


class MossFormerMaskNet(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Args:
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    att_model : torch.nn.module
        Attention model to process the input sequence.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_connection : bool
        Skip connection around attention module.
    use_global_pos_enc : bool
        Global positional encodings.

    Example
    ---------
    >>> mossformer_block = MossFormerM(num_blocks=8, d_model=512)
    >>> mossformer_masknet = MossFormerMaskNet(64, 64, att_model, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = mossformer_masknet(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        att_model,
        norm='ln',
        num_spks=2,
        skip_connection=True,
        use_global_pos_enc=True,
    ):
        super(MossFormerMaskNet, self).__init__()
        self.num_spks = num_spks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(
            in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = copy.deepcopy(
            ComputeAttention(
                att_model,
                out_channels,
                norm,
                skip_connection=skip_connection,
            ))

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(
            out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Args:
            x: Input tensor of dimension [B, N, S].

        Returns:
            out: Output tensor of dimension [spks, B, N, S]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               S = the number of time frames
        """

        # before each line we indicate the shape after executing the line
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d_encoder(x)
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            x = base + emb
        # [B, N, S]
        x = self.mdl(x)
        x = self.prelu(x)
        # [B, N*spks, S]
        x = self.conv1d_out(x)
        b, _, s = x.shape
        # [B*spks, N, S]
        x = x.view(b * self.num_spks, -1, s)
        # [B*spks, N, S]
        x = self.output(x) * self.output_gate(x)
        # [B*spks, N, S]
        x = self.conv1_decoder(x)
        # [B, spks, N, S]
        _, n, L = x.shape
        x = x.view(b, self.num_spks, n, L)
        x = self.activation(x)
        # [spks, B, N, S]
        x = x.transpose(0, 1)
        return x
