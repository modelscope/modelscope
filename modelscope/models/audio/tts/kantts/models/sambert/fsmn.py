# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNet(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, d_out, kernel_size=[1, 1], dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_out,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = F.relu(self.w_1(output))
        output = self.dropout(output)
        output = self.w_2(output)
        output = output.transpose(1, 2)

        return output


class MemoryBlockV2(nn.Module):

    def __init__(self, d, filter_size, shift, dropout=0.0):
        super(MemoryBlockV2, self).__init__()

        left_padding = int(round((filter_size - 1) / 2))
        right_padding = int((filter_size - 1) / 2)
        if shift > 0:
            left_padding += shift
            right_padding -= shift

        self.lp, self.rp = left_padding, right_padding

        self.conv_dw = nn.Conv1d(d, d, filter_size, 1, 0, groups=d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        if mask is not None:
            input = input.masked_fill(mask.unsqueeze(-1), 0)

        x = F.pad(
            input, (0, 0, self.lp, self.rp, 0, 0), mode='constant', value=0.0)
        output = (
            self.conv_dw(x.contiguous().transpose(1,
                                                  2)).contiguous().transpose(
                                                      1, 2))
        output += input
        output = self.dropout(output)

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output


class FsmnEncoderV2(nn.Module):

    def __init__(
        self,
        filter_size,
        fsmn_num_layers,
        input_dim,
        num_memory_units,
        ffn_inner_dim,
        dropout=0.0,
        shift=0,
    ):
        super(FsmnEncoderV2, self).__init__()

        self.filter_size = filter_size
        self.fsmn_num_layers = fsmn_num_layers
        self.num_memory_units = num_memory_units
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.shift = shift
        if not isinstance(shift, list):
            self.shift = [shift for _ in range(self.fsmn_num_layers)]

        self.ffn_lst = nn.ModuleList()
        self.ffn_lst.append(
            FeedForwardNet(
                input_dim, ffn_inner_dim, num_memory_units, dropout=dropout))
        for i in range(1, fsmn_num_layers):
            self.ffn_lst.append(
                FeedForwardNet(
                    num_memory_units,
                    ffn_inner_dim,
                    num_memory_units,
                    dropout=dropout))

        self.memory_block_lst = nn.ModuleList()
        for i in range(fsmn_num_layers):
            self.memory_block_lst.append(
                MemoryBlockV2(num_memory_units, filter_size, self.shift[i],
                              dropout))

    def forward(self, input, mask=None):
        x = F.dropout(input, self.dropout, self.training)
        for (ffn, memory_block) in zip(self.ffn_lst, self.memory_block_lst):
            context = ffn(x)
            memory = memory_block(context, mask)
            memory = F.dropout(memory, self.dropout, self.training)
            if memory.size(-1) == x.size(-1):
                memory += x
            x = memory

        return x
