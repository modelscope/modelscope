# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn

from .feedforward import FeedForward
from .multihead_attention import MultiheadAttention


class TransformerBlock(nn.Module):
    """
    Transformer block module.
    """

    def __init__(self, hidden_dim, num_heads, dropout, attn_dropout,
                 ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=attn_dropout)
        self.attn_norm = nn.LayerNorm(
            normalized_shape=hidden_dim, eps=1e-12, elementwise_affine=True)
        self.ff = FeedForward(
            hidden_dim=hidden_dim,
            inner_dim=4 * hidden_dim,
            dropout=ff_dropout)
        self.ff_norm = nn.LayerNorm(
            normalized_shape=hidden_dim, eps=1e-12, elementwise_affine=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, inp, mask=None, cache=None):
        """Forward process on one transformer layer.

        Args:
            x(Variable(shape: [batch_size, seq_len, hidden_size]))
            memory(Variable(shape: [batch_size, seq_len, hidden_size]))
            mask
            cache
        """
        attn_out = self.attn(inp, mask, cache)
        attn_out = self.dropout_layer(attn_out)
        attn_out = self.attn_norm(attn_out + inp)

        ff_out = self.ff(attn_out)
        ff_out = self.dropout_layer(ff_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out


def main():
    import numpy as np

    model = TransformerBlock(10, 2, 0.5, 0.5, 0.5)
    inp = np.random.rand(2, 3, 10).astype('float32')
    inp = torch.tensor(inp)
    mask = (np.random.rand(2, 3, 3) > 0.5).astype('float32')
    mask = torch.tensor(mask)
    out = model(inp, mask=mask, cache=None)
    print(out)


if __name__ == '__main__':
    main()
