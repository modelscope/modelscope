# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Prenet
from .fsmn import FsmnEncoderV2


class LengthRegulator(nn.Module):

    def __init__(self, r=1):
        super(LengthRegulator, self).__init__()

        self.r = r

    def forward(self, inputs, durations, masks=None):
        reps = (durations + 0.5).long()
        output_lens = reps.sum(dim=1)
        max_len = output_lens.max()
        reps_cumsum = torch.cumsum(
            F.pad(reps.float(), (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
        range_ = torch.arange(max_len).to(inputs.device)[None, :, None]
        mult = (reps_cumsum[:, :, :-1] <= range_) & (
            reps_cumsum[:, :, 1:] > range_)
        mult = mult.float()
        out = torch.matmul(mult, inputs)

        if masks is not None:
            out = out.masked_fill(masks.unsqueeze(-1), 0.0)

        seq_len = out.size(1)
        padding = self.r - int(seq_len) % self.r
        if padding < self.r:
            out = F.pad(
                out.transpose(1, 2), (0, padding, 0, 0, 0, 0), value=0.0)
            out = out.transpose(1, 2)

        return out, output_lens


class VarRnnARPredictor(nn.Module):

    def __init__(self, cond_units, prenet_units, rnn_units):
        super(VarRnnARPredictor, self).__init__()

        self.prenet = Prenet(1, prenet_units)
        self.lstm = nn.LSTM(
            prenet_units[-1] + cond_units,
            rnn_units,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(rnn_units, 1)

    def forward(self, inputs, cond, h=None, masks=None):
        x = torch.cat([self.prenet(inputs), cond], dim=-1)
        # The input can also be a packed variable length sequence,
        # here we just omit it for simplicity due to the mask and uni-directional lstm.
        x, h_new = self.lstm(x, h)

        x = self.fc(x).squeeze(-1)
        x = F.relu(x)

        if masks is not None:
            x = x.masked_fill(masks, 0.0)

        return x, h_new

    def infer(self, cond, masks=None):
        batch_size, length = cond.size(0), cond.size(1)

        output = []
        x = torch.zeros((batch_size, 1)).to(cond.device)
        h = None

        for i in range(length):
            x, h = self.forward(x.unsqueeze(1), cond[:, i:i + 1, :], h=h)
            output.append(x)

        output = torch.cat(output, dim=-1)

        if masks is not None:
            output = output.masked_fill(masks, 0.0)

        return output


class VarFsmnRnnNARPredictor(nn.Module):

    def __init__(
        self,
        in_dim,
        filter_size,
        fsmn_num_layers,
        num_memory_units,
        ffn_inner_dim,
        dropout,
        shift,
        lstm_units,
    ):
        super(VarFsmnRnnNARPredictor, self).__init__()

        self.fsmn = FsmnEncoderV2(
            filter_size,
            fsmn_num_layers,
            in_dim,
            num_memory_units,
            ffn_inner_dim,
            dropout,
            shift,
        )
        self.blstm = nn.LSTM(
            num_memory_units,
            lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * lstm_units, 1)

    def forward(self, inputs, masks=None):
        input_lengths = None
        if masks is not None:
            input_lengths = torch.sum((~masks).float(), dim=1).long()

        x = self.fsmn(inputs, masks)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x,
                input_lengths.tolist(),
                batch_first=True,
                enforce_sorted=False)
            x, _ = self.blstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                x, batch_first=True, total_length=inputs.size(1))
        else:
            x, _ = self.blstm(x)

        x = self.fc(x).squeeze(-1)

        if masks is not None:
            x = x.masked_fill(masks, 0.0)

        return x
