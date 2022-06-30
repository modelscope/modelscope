# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Positional feed forward layer.
    """

    def __init__(self, hidden_dim, inner_dim, dropout):
        super(FeedForward, self).__init__()

        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.linear_hidden = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim), nn.GELU())
        self.linear_out = nn.Linear(inner_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        return

    def forward(self, x):
        out = self.linear_hidden(x)
        out = self.dropout_layer(out)
        out = self.linear_out(out)
        return out


def main():
    import numpy as np

    model = FeedForward(10, 20, 0.5)
    inp = np.random.rand(2, 3, 10).astype('float32')
    inp = torch.tensor(inp)
    out = model(inp)
    print(out)


if __name__ == '__main__':
    main()
