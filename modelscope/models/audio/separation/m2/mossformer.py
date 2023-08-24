# Copyright (c) Alibaba, Inc. and its affiliates.
# Some code here is modified based on speechbrain and can be found on github
# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py
"""Library to support dual-path speech separation.

Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.utils.constant import ModelFile, Tasks
from .mossformer_block import MossformerBlockGFSMN, ScaledSinuEmbedding

EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Args:
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example:
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        """Returns the normalized tensor.

        Args:
            x : torch.Tensor
                Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                # yapf: disable
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps)
                     + self.bias)
                # yapf: enable
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                # yapf: disable
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps)
                     + self.bias)
                # yapf: enable
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

    Args:
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


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
        kernel_size : int
            Length of filters.
        in_channels : int
            Number of  input channels.
        out_channels : int
            Number of output channels.

    Example:
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
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

    def forward(self, x):
        """Return the encoded output.

        Args:
            x : torch.Tensor
                Input tensor with dimensionality [B, L].

        Returns:
            x : torch.Tensor
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
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example:
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
            x : torch.Tensor
                Input tensor with dimensionality [B, N, L].
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

    Example:
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder_MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(self,
                 num_blocks,
                 d_model=None,
                 causal=False,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=4.,
                 attn_dropout=0.1):
        super().__init__()

        self.mossformerM = MossformerBlockGFSMN(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        """
        Args:
            src : torch.Tensor
                Tensor shape [B, L, N],
                where, B = Batchsize,
                       L = time points
                       N = number of filters
                The sequence to the encoder layer (required).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output


class ComputationBlock(nn.Module):
    """Computation block for dual-path processing.

    Args:
        num_blocks : int
            Number of mossformer blocks to include.
         out_channels : int
            Dimensionality of inter/intra model.
         norm : str
            Normalization type.
         skip_around_intra : bool
            Skip connection around the intra layer.

    Example:
    ---------
        >>> comp_block = ComputationBlock(64)
        >>> x = torch.randn(10, 64, 100)
        >>> x = comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100])
    """

    def __init__(
        self,
        num_blocks,
        out_channels,
        norm='ln',
        skip_around_intra=True,
    ):
        super(ComputationBlock, self).__init__()

        # MossFormer+: MossFormer with recurrence
        self.intra_mdl = MossFormerM(
            num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def forward(self, x):
        """Returns the output tensor.

        Args:
            x : torch.Tensor
                Input tensor of dimension [B, N, S].

        Returns:
            out: torch.Tensor
                Output tensor of dimension [B, N, S].
                where, B = Batchsize,
                   N = number of filters
                   S = sequence time index
        """
        B, N, S = x.shape
        # intra RNN
        # [B, S, N]
        intra = x.permute(0, 2, 1).contiguous()

        intra = self.intra_mdl(intra)

        # [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, S]
        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out


class MossFormerMaskNet(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Args:
        in_channels : int
            Number of channels at the output of the encoder.
        out_channels : int
            Number of channels that would be inputted to the intra and inter blocks.
        norm : str
            Normalization type.
        num_spks : int
            Number of sources (speakers).
        skip_around_intra : bool
            Skip connection around intra.
        use_global_pos_enc : bool
            Global positional encodings.
        max_length : int
            Maximum sequence length.

    Example:
    ---------
    >>> mossformer_block = MossFormerM(1, 64, 8)
    >>> mossformer_masknet = MossFormerMaskNet(64, 64, intra_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = mossformer_masknet(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=24,
        norm='ln',
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormerMaskNet, self).__init__()
        self.num_spks = num_spks
        self.num_blocks = num_blocks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(
            in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = ComputationBlock(
            num_blocks,
            out_channels,
            norm,
            skip_around_intra=skip_around_intra,
        )

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

    def forward(self, x):
        """Returns the output tensor.

        Args:
            x : torch.Tensor
                Input tensor of dimension [B, N, S].

        Returns:
            out : torch.Tensor
                Output tensor of dimension [spks, B, N, S]
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
        B, _, S = x.shape

        # [B*spks, N, S]
        x = x.view(B * self.num_spks, -1, S)

        # [B*spks, N, S]
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, S]
        x = self.conv1_decoder(x)

        # [B, spks, N, S]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, S]
        x = x.transpose(0, 1)

        return x


@MODELS.register_module(
    Tasks.speech_separation,
    module_name=Models.speech_mossformer2_separation_temporal_8k)
class MossFormer2(TorchModel):
    """Library to support MossFormer speech separation.

    Args:
        model_dir (str): the model path.
    """

    def __init__(self,
                 model_dir: str,
                 in_channels=512,
                 out_channels=512,
                 num_blocks=24,
                 kernel_size=16,
                 norm='ln',
                 num_spks=2,
                 skip_around_intra=True,
                 use_global_pos_enc=True,
                 max_length=20000,
                 *args,
                 **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.num_spks = num_spks
        self.enc = Encoder(
            kernel_size=kernel_size, out_channels=in_channels, in_channels=1)
        self.mask_net = MossFormerMaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        self.dec = Decoder(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False)

    def forward(self, input):
        x = self.enc(input)
        mask = self.mask_net(x)
        x = torch.stack([x] * self.num_spks)
        sep_x = x * mask

        # Decoding
        est_source = torch.cat(
            [self.dec(sep_x[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        return est_source

    def load_check_point(self, load_path=None, device=None):
        if not load_path:
            load_path = self.model_dir
        if not device:
            device = torch.device('cpu')
        self.load_state_dict(
            torch.load(
                os.path.join(load_path, ModelFile.TORCH_MODEL_FILE),
                map_location=device),
            strict=False)
