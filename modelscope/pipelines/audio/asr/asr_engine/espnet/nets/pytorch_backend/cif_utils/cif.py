# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
import logging

import numpy as np
import torch
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from torch import nn


class CIF_Model(nn.Module):

    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1):
        super(CIF_Model, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(
            idim, idim, l_order + r_order + 1, groups=idim)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        memory = self.cif_conv1d(queries)
        output = memory + context
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        if mask is not None:
            alphas = alphas * mask.transpose(-1, -2).float()
        alphas = alphas.squeeze(-1)
        if target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        cif_length = alphas.sum(-1)
        if target_label is not None:
            alphas *= (target_length / cif_length)[:, None].repeat(
                1, alphas.size(1))
        cif_output, cif_peak = cif(hidden, alphas, self.threshold)
        return cif_output, cif_length, target_length, cif_peak

    def gen_frame_alignments(self,
                             alphas: torch.Tensor = None,
                             memory_sequence_length: torch.Tensor = None,
                             is_training: bool = True,
                             dtype: torch.dtype = torch.float32):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32
        token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = torch.tile(alphas_cumsum[:, None, :],
                                   [1, max_token_num, 1])

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = torch.tile(index[:, :, None], [1, 1, maximum_length])

        index_div = torch.floor(torch.divide(alphas_cumsum,
                                             index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(
            index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clip(index_div_bool_zeros_count, 0,
                                                memory_sequence_length.max())
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(
            token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = torch.tile(
            index_div_bool_zeros_count[:, :, None], [1, 1, maximum_length])
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(
            torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(
            int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(
            index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(
            int_type)
        predictor_mask = (~make_pad_mask(
            memory_sequence_length,
            maxlen=memory_sequence_length.max())).type(int_type).to(
                memory_sequence_length.device)  # noqa: *
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask
        return index_div_bool_zeros_count_tile_out.detach(
        ), index_div_bool_zeros_count.detach()


class cif_predictor(nn.Module):

    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1):
        super(cif_predictor, self).__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(
            idim, idim, l_order + r_order + 1, groups=idim)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold

    def forward(self,
                hidden,
                target_label=None,
                mask=None,
                ignore_id=-1,
                mask_chunk_predictor=None,
                target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        memory = self.cif_conv1d(queries)
        output = memory + context
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        if mask is not None:
            alphas = alphas * mask.transpose(-1, -2).float()
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(
                1, alphas.size(1))
        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        return acoustic_embeds, token_num, alphas, cif_peak

    def gen_frame_alignments(self,
                             alphas: torch.Tensor = None,
                             memory_sequence_length: torch.Tensor = None,
                             is_training: bool = True,
                             dtype: torch.dtype = torch.float32):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32
        token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)

        max_token_num = torch.max(token_num).item()

        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = torch.tile(alphas_cumsum[:, None, :],
                                   [1, max_token_num, 1])

        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = torch.tile(index[:, :, None], [1, 1, maximum_length])

        index_div = torch.floor(torch.divide(alphas_cumsum,
                                             index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(
            index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clip(index_div_bool_zeros_count, 0,
                                                memory_sequence_length.max())
        token_num_mask = (~make_pad_mask(token_num, maxlen=max_token_num)).to(
            token_num.device)
        index_div_bool_zeros_count *= token_num_mask

        index_div_bool_zeros_count_tile = torch.tile(
            index_div_bool_zeros_count[:, :, None], [1, 1, maximum_length])
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)

        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(
            torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(
            int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(
            index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(
            int_type)
        predictor_mask = (~make_pad_mask(
            memory_sequence_length,
            maxlen=memory_sequence_length.max())).type(int_type).to(
                memory_sequence_length.device)  # noqa: *
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask
        return index_div_bool_zeros_count_tile_out.detach(
        ), index_div_bool_zeros_count.detach()


def cif(hidden, alphas, threshold):
    batch_size, len_time, hidden_size = hidden.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size],
                                             device=hidden.device) - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], device=hidden.device),
            integrate)
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                            remainds[:, None] * hidden[:, t, :], frame)

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        ls = torch.index_select(frames[b, :, :], 0,
                                torch.nonzero(fire >= threshold).squeeze())
        pad_l = torch.zeros([max_label_len - ls.size(0), hidden_size],
                            device=hidden.device)
        list_ls.append(torch.cat([ls, pad_l], 0))
    return torch.stack(list_ls, 0), fires
