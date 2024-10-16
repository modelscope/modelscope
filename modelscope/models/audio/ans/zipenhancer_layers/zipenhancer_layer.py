#!/usr/bin/env python3
#
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
from typing import List, Optional, Tuple, Union

import torch
from .scaling import (
    FloatLike,
    ScheduledFloat,
    convert_num_channels,
)
from torch import Tensor, nn

from .zipformer import SimpleDownsample, SimpleUpsample, CompactRelPositionalEncoding, BypassModule
from .zipformer import Zipformer2EncoderLayer




class DualPathZipformer2Encoder(nn.Module):
    r"""DualPathZipformer2Encoder is a stack of N encoder layers
    it has two kinds of EncoderLayer including F_Zipformer2EncoderLayer and T_Zipformer2EncoderLayer
    the features are modeling with the shape of
    [B, C, T, F] -> [F, T * B, C] -> -> [B, C, T, F] -> [T, F * B, C] -> [B, C, T, F]

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> dualpath_zipformer_encoder = DualPathZipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 512, 161, 101)
        >>> out = dualpath_zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
        bypass_layer=None,
    ) -> None:
        """
        Initialize the DualPathZipformer2Encoder module with the specified
        encoder layer, number of layers, positional dimension, dropout rate, warmup period, and layer drop rates.
        """
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(
            pos_dim, dropout_rate=0.15, length_factor=1.0
        )

        self.f_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.t_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.bypass_layers = nn.ModuleList(
            [bypass_layer for i in range(num_layers * 2)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.f_layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            self.t_layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in a dual-path manner, processing both temporal and frequency dimensions.

        Args:
            src: the dual-path sequence to the encoder (required): shape (batch_size, embedding_dim, seq_len, frequency_len).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking. No used.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """

        # src: (b, c, t, f)
        b, c, t, f = src.size()
        src_f = src.permute(3, 0, 2, 1).contiguous().view(f, b * t, c)
        src_t = src.permute(2, 0, 3, 1).contiguous().view(t, b * f, c)
        pos_emb_f = self.encoder_pos(src_f)
        pos_emb_t = self.encoder_pos(src_t)

        output = src

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            output = output * feature_mask

        for i in range(len(self.f_layers)):
            # output_org = output
            # (b, c, t, f)
            output_f_org = output.permute(3, 2, 0, 1).contiguous() # (f, t, b, c)
            output_f = output_f_org.view(f, t * b, c)
            # (f, t * b, c)
            output_f = self.f_layers[i](
                output_f,
                pos_emb_f,
                # chunk_size=chunk_size,
                # attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            output_f = output_f.view(f, t, b, c)
            output_f = self.bypass_layers[i * 2](output_f_org, output_f)

            # (f, t, b, c)
            output = output_f.permute(2, 3, 1, 0).contiguous()
            # (b, c, t, f)
            # output = self.bypass_layers[i * 2](output_org, output)

            # output_org = output

            output_t_org = output.permute(2, 3, 0, 1).contiguous() # (t, f, b, c)
            output_t = output_t_org.view(t, f * b, c)
            output_t = self.t_layers[i](
                output_t,
                pos_emb_t,
                # chunk_size=chunk_size,
                # attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            output_t = output_t.view(t, f, b, c)
            output_t = self.bypass_layers[i * 2 + 1](output_t_org, output_t)
            # (t, f, b, c)

            output = output_t.permute(2, 3, 0, 1).contiguous()
            # (b, c, t, f)
            # output = self.bypass_layers[i * 2 + 1](output_org, output)

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                output = output * feature_mask


        return output


class DualPathDownsampledZipformer2Encoder(nn.Module):
    r"""
    DualPathDownsampledZipformer2Encoder is a dual-path zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    The features are downsampled-upsampled at the time and frequency domain.

    """

    def __init__(
        self, encoder: nn.Module, dim: int, t_downsample: int, f_downsample: int, dropout: FloatLike
    ):
        """
        Initialize the DualPathDownsampledZipformer2Encoder module with the specified
        encoder, dimension, temporal and frequency downsampling factors r, and dropout rate.
        """
        super(DualPathDownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = t_downsample
        self.t_downsample_factor = t_downsample
        self.f_downsample_factor = f_downsample

        if self.t_downsample_factor != 1:
            self.downsample_t = SimpleDownsample(dim, t_downsample, dropout)
            self.upsample_t = SimpleUpsample(dim, t_downsample)
        if self.f_downsample_factor != 1:
            self.downsample_f = SimpleDownsample(dim, f_downsample, dropout)
            self.upsample_f = SimpleUpsample(dim, f_downsample)

        # self.num_layers = encoder.num_layers
        self.encoder = encoder


        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample the input, process through the encoder, and then upsample back to the original shape.

        Args:
            src: the sequence to the encoder (required): shape (batch_size, embedding_dim, seq_len, frequency_len).
            feature_mask: 1.0
            attn_mask: None
            src_key_padding_mask: None.

        Returns: a Tensor with the same shape as src. (batch_size, embedding_dim, seq_len, frequency_len)
        """
        # src: (b, c, t, f)
        b, c, t, f = src.size()
        # print(src.size())

        src_orig = src.permute(2, 3, 0, 1) # (t, f, b, c)

        # (b, c, t, f)
        src = src.permute(2, 0, 3, 1).contiguous().view(t, b * f, c)
        # -> (t, b * f, c)
        if self.t_downsample_factor != 1:
            src = self.downsample_t(src)
        # (t//ds + 1, b * f, c)
        downsample_t = src.size(0)
        src = src.view(downsample_t, b, f, c).permute(2, 1, 0, 3).contiguous().view(f, b * downsample_t, c)
        # src = self.upsample_f(src)
        if self.f_downsample_factor != 1:
            src = self.downsample_f(src)
        # (f//ds + 1, b * downsample_t, c)
        downsample_f = src.size(0)
        src = src.view(downsample_f, b, downsample_t, c).permute(1, 3, 2, 0)
        # (b, c, downsample_t, downsample_f)
        # print(src.size())


        # ds = self.downsample_factor
        # if attn_mask is not None:
        #     attn_mask = attn_mask[::ds, ::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # (b, c, downsample_t, downsample_f)
        src = src.permute(3, 0, 2, 1).contiguous().view(downsample_f, b * downsample_t, c)
        if self.f_downsample_factor != 1:
            src = self.upsample_f(src)
        # (f, b * downsample_t, c)
        src = src[: f].view(f, b, downsample_t, c).permute(2, 1, 0, 3).contiguous().view(downsample_t, b * f, c)
        # (downsample_t, b * f, c)
        if self.t_downsample_factor != 1:
            src = self.upsample_t(src)
        # (t, b * f, c)
        src = src[: t].view(t, b, f, c).permute(0, 2, 1, 3).contiguous()
        # (t, f, b, c)
        out = self.out_combiner(src_orig, src)
        # (t, f, b, c)

        out = out.permute(2, 3, 0, 1).contiguous()
        # (b, c, t, f)
        # print(out.size())

        # remove any extra frames that are not a multiple of downsample_factor
        # src = src[: src_orig.shape[0]] # slice here

        return out




class Zipformer2DualPathEncoder(nn.Module):
    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        f_downsampling_factor: Tuple[int] = None,
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ):
        """
        Initialize the Zipformer2DualPathEncoder module.
        Zipformer2DualPathEncoder processes the hidden features of the noisy speech using dual-path modeling.
        It has two kinds of blocks: DualPathZipformer2Encoder and DualPathDownsampledZipformer2Encoder.
        DualPathZipformer2Encoder processes the 4D features with the shape of [B, C, T, F].
        DualPathDownsampledZipformer2Encoder first downsamples the hidden features and processes features using dual-path modeling like DualPathZipformer2Encoder.

        Args:
        Various hyperparameters and settings for the encoder.
        """
        super(Zipformer2DualPathEncoder, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        def _to_tuple(x):
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor  # int
        self.downsampling_factor = downsampling_factor  # tuple

        if f_downsampling_factor is None:
            f_downsampling_factor = downsampling_factor
        self.f_downsampling_factor = _to_tuple(f_downsampling_factor)

        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)  # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(
            encoder_unmasked_dim
        )  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u, d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        # "1,2,4,8,4,2",

        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = DualPathZipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
                bypass_layer=BypassModule(encoder_dim[i], straight_through_rate=0),
            )

            if downsampling_factor[i] != 1 or f_downsampling_factor[i] != 1:
                encoder = DualPathDownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    t_downsample=downsampling_factor[i],
                    f_downsample=f_downsampling_factor[i],
                    dropout=dropout,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(
            max(encoder_dim), downsample=output_downsampling_factor, dropout=dropout
        )

    def forward(self, x):
        """
        Forward pass of the Zipformer2DualPathEncoder module.

        Args:
        x (Tensor): Input tensor of shape [B, C, T, F].

        Returns:
        Tensor: Output tensor after passing through the encoder.
        """
        outputs = []

        # if torch.jit.is_scripting() or torch.jit.is_tracing():
        #     feature_masks = [1.0] * len(self.encoder_dim)
        # else:
            # feature_masks = self.get_feature_masks(x)
        feature_masks = [1.0] * len(self.encoder_dim)
        attn_mask = None

        chunk_size, left_context_chunks = -1, -1

        for i, module in enumerate(self.encoders):

            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(
                x,
                chunk_size=chunk_size,
                feature_mask=feature_masks[i],
                src_key_padding_mask=None,
                attn_mask=attn_mask,
            )
            outputs.append(x)

        # (b, c, t, f)
        return x


if __name__ == "__main__":

    # {2,2,2,2,2,2} {192,256,256,256,256,256} {512,768,768,768,768,768}
    downsampling_factor = (1, 2, 4, 3)  #
    encoder_dim = (16, 32, 64, 64)
    pos_dim = 48  # zipformer base设置
    num_heads = (4, 4, 4, 4)  # "4,4,4,8,4,4"
    query_head_dim = (16,) * len(downsampling_factor)  # 32
    pos_head_dim = (4,) * len(downsampling_factor)  # 4
    value_head_dim = (12,) * len(downsampling_factor)  # 12
    feedforward_dim = (32, 64, 128, 128)  #
    dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))
    cnn_module_kernel = (15,) * len(downsampling_factor)  # 31,31,15,15,15,31
    causal = False
    encoder_unmasked_dim = (16, ) * len(downsampling_factor)

    num_encoder_layers = (1, 1, 1, 1)
    warmup_batches = 4000.0

    net = Zipformer2DualPathEncoder(
        output_downsampling_factor = 1,
        downsampling_factor=downsampling_factor,
        num_encoder_layers=num_encoder_layers,
        encoder_dim=encoder_dim,
        encoder_unmasked_dim=encoder_unmasked_dim,
        query_head_dim=query_head_dim,
        pos_head_dim=pos_head_dim,
        value_head_dim=value_head_dim,
        pos_dim=pos_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        cnn_module_kernel=cnn_module_kernel,
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=warmup_batches,
        causal=causal,
    )

    # net = DownsampledZipformer2Encoder(
    #     None, 128, 2, 0.
    # )
    # x = torch.randn((101, 2, 128))
    b = 4
    t = 321
    f = 101
    c = 64



    # x = torch.randn((101, 2, 128))
    x = torch.randn((b, c, t, f))

    x = net(x)
    print(x.size())