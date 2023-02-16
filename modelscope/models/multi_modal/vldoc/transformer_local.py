# The implementation is borrowed and modified from the official PyTorch website and ABINet:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
# https://github.com/FangShancheng/ABINet/blob/main/modules/transformer.py

import copy

import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
from torch.nn import functional as F


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt,
                memory,
                memory2=None,
                tgt_mask=None,
                memory_mask=None,
                memory_mask2=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                memory_key_padding_mask2=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                memory2=memory2,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                memory_mask2=memory_mask2,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_key_padding_mask2=memory_key_padding_mask2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 self_attn=True,
                 siamese=False,
                 debug=False):
        super(TransformerDecoderLayer, self).__init__()
        self.has_self_attn, self.siamese = self_attn, siamese
        self.debug = debug
        if self.has_self_attn:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout)
            self.norm1 = LayerNorm(d_model)
            self.dropout1 = Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        if self.siamese:
            self.multihead_attn2 = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                memory2=None,
                memory_mask2=None,
                memory_key_padding_mask2=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.has_self_attn:
            tgt2, attn = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            if self.debug:
                self.attn = attn
        tgt2, attn2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        if self.debug:
            self.attn2 = attn2

        if self.siamese:
            tgt3, attn3 = self.multihead_attn2(
                tgt,
                memory2,
                memory2,
                attn_mask=memory_mask2,
                key_padding_mask=memory_key_padding_mask2)
            tgt = tgt + self.dropout2(tgt3)
            if self.debug:
                self.attn3 = attn3

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError(
        'activation should be relu/gelu, not {}'.format(activation))
