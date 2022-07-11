# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
"""Transformer encoder definition."""

import logging
from typing import List, Optional, Sequence, Tuple, Union

import torch
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import \
    PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear, MultiLayeredConv1d)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import \
    PositionwiseFeedForward  # noqa: H301
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6,
    Conv2dSubsampling8, TooShortUttError, check_short_utt)
from typeguard import check_argument_types

from ...asr.streaming_utilis.chunk_utilis import overlap_chunk
from ...nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention, MultiHeadedAttentionSANM)
from ...nets.pytorch_backend.transformer.encoder_layer import (
    EncoderLayer, EncoderLayerChunk)


class SANMEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = 'conv2d',
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = 'linear',
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = 'sanm',
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if input_layer == 'linear':
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == 'conv2d':
            self.embed = Conv2dSubsampling(input_size, output_size,
                                           dropout_rate)
        elif input_layer == 'conv2d2':
            self.embed = Conv2dSubsampling2(input_size, output_size,
                                            dropout_rate)
        elif input_layer == 'conv2d6':
            self.embed = Conv2dSubsampling6(input_size, output_size,
                                            dropout_rate)
        elif input_layer == 'conv2d8':
            self.embed = Conv2dSubsampling8(input_size, output_size,
                                            dropout_rate)
        elif input_layer == 'embed':
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(
                    input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError('unknown input_layer: ' + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == 'linear':
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == 'conv1d':
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == 'conv1d-linear':
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError('Support only linear or conv1d.')

        if selfattention_layer_type == 'selfattn':
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == 'sanm':
            encoder_selfattn_layer = MultiHeadedAttentionSANM
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(
                interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (isinstance(self.embed, Conv2dSubsampling)
              or isinstance(self.embed, Conv2dSubsampling2)
              or isinstance(self.embed, Conv2dSubsampling6)
              or isinstance(self.embed, Conv2dSubsampling8)):
            short_status, limit_size = check_short_utt(self.embed,
                                                       xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f'has {xs_pad.size(1)} frames and is too short for subsampling '
                    +  # noqa: *
                    f'(it needs more than {limit_size} frames), return empty results',
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None


class SANMEncoderChunk(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = 'conv2d',
            pos_enc_class=PositionalEncoding,
            normalize_before: bool = True,
            concat_after: bool = False,
            positionwise_layer_type: str = 'linear',
            positionwise_conv_kernel_size: int = 1,
            padding_idx: int = -1,
            interctc_layer_idx: List[int] = [],
            interctc_use_conditioning: bool = False,
            kernel_size: int = 11,
            sanm_shfit: int = 0,
            selfattention_layer_type: str = 'sanm',
            chunk_size: Union[int, Sequence[int]] = (16, ),
            stride: Union[int, Sequence[int]] = (10, ),
            pad_left: Union[int, Sequence[int]] = (0, ),
            encoder_att_look_back_factor: Union[int, Sequence[int]] = (1, ),
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if input_layer == 'linear':
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == 'conv2d':
            self.embed = Conv2dSubsampling(input_size, output_size,
                                           dropout_rate)
        elif input_layer == 'conv2d2':
            self.embed = Conv2dSubsampling2(input_size, output_size,
                                            dropout_rate)
        elif input_layer == 'conv2d6':
            self.embed = Conv2dSubsampling6(input_size, output_size,
                                            dropout_rate)
        elif input_layer == 'conv2d8':
            self.embed = Conv2dSubsampling8(input_size, output_size,
                                            dropout_rate)
        elif input_layer == 'embed':
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(
                    input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError('unknown input_layer: ' + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == 'linear':
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == 'conv1d':
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == 'conv1d-linear':
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError('Support only linear or conv1d.')

        if selfattention_layer_type == 'selfattn':
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == 'sanm':
            encoder_selfattn_layer = MultiHeadedAttentionSANM
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                kernel_size,
                sanm_shfit,
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayerChunk(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(
                interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        shfit_fsmn = (kernel_size - 1) // 2
        self.overlap_chunk_cls = overlap_chunk(
            chunk_size=chunk_size,
            stride=stride,
            pad_left=pad_left,
            shfit_fsmn=shfit_fsmn,
            encoder_att_look_back_factor=encoder_att_look_back_factor,
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        ind: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (isinstance(self.embed, Conv2dSubsampling)
              or isinstance(self.embed, Conv2dSubsampling2)
              or isinstance(self.embed, Conv2dSubsampling6)
              or isinstance(self.embed, Conv2dSubsampling8)):
            short_status, limit_size = check_short_utt(self.embed,
                                                       xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f'has {xs_pad.size(1)} frames and is too short for subsampling '
                    +  # noqa: *
                    f'(it needs more than {limit_size} frames), return empty results',
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        mask_shfit_chunk, mask_att_chunk_encoder = None, None
        if self.overlap_chunk_cls is not None:
            ilens = masks.squeeze(1).sum(1)
            chunk_outs = self.overlap_chunk_cls.gen_chunk_mask(ilens, ind)
            xs_pad, ilens = self.overlap_chunk_cls.split_chunk(
                xs_pad, ilens, chunk_outs=chunk_outs)
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
            mask_shfit_chunk = self.overlap_chunk_cls.get_mask_shfit_chunk(
                chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype)
            mask_att_chunk_encoder = self.overlap_chunk_cls.get_mask_att_chunk_encoder(
                chunk_outs, xs_pad.device, xs_pad.size(0), dtype=xs_pad.dtype)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks, _, _, _ = self.encoders(xs_pad, masks, None,
                                                   mask_shfit_chunk,
                                                   mask_att_chunk_encoder)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks, _, _, _ = encoder_layer(xs_pad, masks, None,
                                                       mask_shfit_chunk,
                                                       mask_att_chunk_encoder)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if self.overlap_chunk_cls is not None:
            xs_pad, olens = self.overlap_chunk_cls.remove_chunk(
                xs_pad, ilens, chunk_outs)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
