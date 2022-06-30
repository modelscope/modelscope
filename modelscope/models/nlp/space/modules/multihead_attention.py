# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """
    Multi head attention layer.
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        assert hidden_dim % num_heads == 0
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.linear_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        return

    def _split_heads(self, x, is_key=False):
        x = x.reshape(x.size(0), x.size(1), self.num_heads, self.head_dim)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), self.hidden_dim)
        return x

    def _attn(self, query, key, value, mask):
        # shape: [batch_size, num_head, seq_len, seq_len]
        scores = torch.matmul(query, key)
        scores = scores * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(
                mask.bool(),
                float('-inf'))  # scores = (1 - mask) * scores + mask * (-1e10)

        attn = self.softmax(scores)
        attn = self.dropout_layer(attn)

        if mask is not None:
            '''
            mask: [batch size, num_heads, seq_len, seq_len]

            >>> F.softmax([-1e10, -100, -100])
            >>> [0.00, 0.50, 0.50]
            >>> F.softmax([-1e10, -1e10, -1e10])
            >>> [0.33, 0.33, 0.33]
            ==> [0.00, 0.00, 0.00]
            '''
            attn.masked_fill_(mask.bool(), 0.)  # attn = (1 - mask) * attn

        out = torch.matmul(attn, value)
        return out

    def forward(self, inp, mask=None, cache=None):
        """ Forward process of self attention. """
        # shape: [batch_size, seq_len, 3 * hidden_dim]
        qkv = self.linear_qkv(inp)
        query, key, value = torch.split(qkv, self.hidden_dim, dim=2)

        # shape: [batch_size, num_head, seq_len, head_dim]
        query = self._split_heads(query)
        # shape: [batch_size, num_head, head_dim, seq_len]
        key = self._split_heads(key, is_key=True)
        # shape: [batch_size, num_head, seq_len, head_dim]
        value = self._split_heads(value)

        if cache is not None:
            if 'key' in cache and 'value' in cache:
                key = torch.cat([cache['key'], key], dim=3)
                value = torch.cat([cache['value'], value], dim=2)
            cache['key'] = key
            cache['value'] = value

        out = self._attn(query, key, value, mask)
        out = self._merge_heads(out)
        out = self.linear_out(out)
        return out


def main():
    import numpy as np

    model = MultiheadAttention(10, 2, 0.5)
    inp = np.random.rand(2, 3, 10).astype('float32')
    inp = torch.tensor(inp)
    mask = (np.random.rand(2, 3, 3) > 0.5).astype('float32')
    mask = torch.tensor(mask)
    out = model(inp, mask=mask, cache=None)
    print(out)


if __name__ == '__main__':
    main()
