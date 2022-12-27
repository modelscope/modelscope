# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoder(nn.Module):

    def __init__(self, max_len, depth):
        super(SinusoidalPositionEncoder, self).__init__()

        self.max_len = max_len
        self.depth = depth
        self.position_enc = nn.Parameter(
            self.get_sinusoid_encoding_table(max_len, depth).unsqueeze(0),
            requires_grad=False,
        )

    def forward(self, input):
        bz_in, len_in, _ = input.size()
        if len_in > self.max_len:
            self.max_len = len_in
            self.position_enc.data = (
                self.get_sinusoid_encoding_table(
                    self.max_len, self.depth).unsqueeze(0).to(input.device))

        output = input + self.position_enc[:, :len_in, :].expand(bz_in, -1, -1)

        return output

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
        """ Sinusoid position encoding table """

        def cal_angle(position, hid_idx):
            return position / np.power(10000, hid_idx / float(d_hid / 2 - 1))

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid // 2)]

        scaled_time_table = np.array(
            [get_posi_angle_vec(pos_i + 1) for pos_i in range(n_position)])

        sinusoid_table = np.zeros((n_position, d_hid))
        sinusoid_table[:, :d_hid // 2] = np.sin(scaled_time_table)
        sinusoid_table[:, d_hid // 2:] = np.cos(scaled_time_table)

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.0

        return torch.FloatTensor(sinusoid_table)


class DurSinusoidalPositionEncoder(nn.Module):

    def __init__(self, depth, outputs_per_step):
        super(DurSinusoidalPositionEncoder, self).__init__()

        self.depth = depth
        self.outputs_per_step = outputs_per_step

        inv_timescales = [
            np.power(10000, 2 * (hid_idx // 2) / depth)
            for hid_idx in range(depth)
        ]
        self.inv_timescales = nn.Parameter(
            torch.FloatTensor(inv_timescales), requires_grad=False)

    def forward(self, durations, masks=None):
        reps = (durations + 0.5).long()
        output_lens = reps.sum(dim=1)
        max_len = output_lens.max()
        reps_cumsum = torch.cumsum(
            F.pad(reps.float(), (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
        range_ = torch.arange(max_len).to(durations.device)[None, :, None]
        mult = (reps_cumsum[:, :, :-1] <= range_) & (
            reps_cumsum[:, :, 1:] > range_)
        mult = mult.float()
        offsets = torch.matmul(mult,
                               reps_cumsum[:,
                                           0, :-1].unsqueeze(-1)).squeeze(-1)
        dur_pos = range_[:, :, 0] - offsets + 1

        if masks is not None:
            assert masks.size(1) == dur_pos.size(1)
            dur_pos = dur_pos.masked_fill(masks, 0.0)

        seq_len = dur_pos.size(1)
        padding = self.outputs_per_step - int(seq_len) % self.outputs_per_step
        if padding < self.outputs_per_step:
            dur_pos = F.pad(dur_pos, (0, padding, 0, 0), value=0.0)

        position_embedding = dur_pos[:, :, None] / self.inv_timescales[None,
                                                                       None, :]
        position_embedding[:, :, 0::2] = torch.sin(position_embedding[:, :,
                                                                      0::2])
        position_embedding[:, :, 1::2] = torch.cos(position_embedding[:, :,
                                                                      1::2])

        return position_embedding
