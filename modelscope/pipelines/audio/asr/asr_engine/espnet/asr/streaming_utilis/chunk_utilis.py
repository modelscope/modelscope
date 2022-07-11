# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from espnet/espnet.
import logging
import math

import numpy as np
import torch
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from ...nets.pytorch_backend.cif_utils.cif import \
    cif_predictor as cif_predictor

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile='full', precision=100000, linewidth=None)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device='cpu'):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype).to(device)


class overlap_chunk():

    def __init__(
            self,
            chunk_size: tuple = (16, ),
            stride: tuple = (10, ),
            pad_left: tuple = (0, ),
            encoder_att_look_back_factor: tuple = (1, ),
            shfit_fsmn: int = 0,
    ):
        self.chunk_size, self.stride, self.pad_left, self.encoder_att_look_back_factor \
            = chunk_size, stride, pad_left, encoder_att_look_back_factor
        self.shfit_fsmn = shfit_fsmn
        self.x_add_mask = None
        self.x_rm_mask = None
        self.x_len = None
        self.mask_shfit_chunk = None
        self.mask_chunk_predictor = None
        self.mask_att_chunk_encoder = None
        self.mask_shift_att_chunk_decoder = None
        self.chunk_size_cur, self.stride_cur, self.pad_left_cur, self.encoder_att_look_back_factor_cur \
            = None, None, None, None

    def get_chunk_size(self, ind: int = 0):
        # with torch.no_grad:
        chunk_size, stride, pad_left, encoder_att_look_back_factor = self.chunk_size[
            ind], self.stride[ind], self.pad_left[
                ind], self.encoder_att_look_back_factor[ind]
        self.chunk_size_cur, self.stride_cur, self.pad_left_cur,
        self.encoder_att_look_back_factor_cur, self.chunk_size_pad_shift_cur \
            = chunk_size, stride, pad_left, encoder_att_look_back_factor, chunk_size + self.shfit_fsmn
        return self.chunk_size_cur, self.stride_cur, self.pad_left_cur, self.encoder_att_look_back_factor_cur

    def gen_chunk_mask(self, x_len, ind=0, num_units=1, num_units_predictor=1):

        with torch.no_grad():
            x_len = x_len.cpu().numpy()
            x_len_max = x_len.max()

            chunk_size, stride, pad_left, encoder_att_look_back_factor = self.get_chunk_size(
                ind)
            shfit_fsmn = self.shfit_fsmn
            chunk_size_pad_shift = chunk_size + shfit_fsmn

            chunk_num_batch = np.ceil(x_len / stride).astype(np.int32)
            x_len_chunk = (
                chunk_num_batch - 1
            ) * chunk_size_pad_shift + shfit_fsmn + pad_left + 0 + x_len - (
                chunk_num_batch - 1) * stride
            x_len_chunk = x_len_chunk.astype(x_len.dtype)
            x_len_chunk_max = x_len_chunk.max()

            chunk_num = int(math.ceil(x_len_max / stride))
            dtype = np.int32
            max_len_for_x_mask_tmp = max(chunk_size, x_len_max)
            x_add_mask = np.zeros([0, max_len_for_x_mask_tmp], dtype=dtype)
            x_rm_mask = np.zeros([max_len_for_x_mask_tmp, 0], dtype=dtype)
            mask_shfit_chunk = np.zeros([0, num_units], dtype=dtype)
            mask_chunk_predictor = np.zeros([0, num_units_predictor],
                                            dtype=dtype)
            mask_shift_att_chunk_decoder = np.zeros([0, 1], dtype=dtype)
            mask_att_chunk_encoder = np.zeros(
                [0, chunk_num * chunk_size_pad_shift], dtype=dtype)
            for chunk_ids in range(chunk_num):
                # x_mask add
                fsmn_padding = np.zeros((shfit_fsmn, max_len_for_x_mask_tmp),
                                        dtype=dtype)
                x_mask_cur = np.diag(np.ones(chunk_size, dtype=np.float32))
                x_mask_pad_left = np.zeros((chunk_size, chunk_ids * stride),
                                           dtype=dtype)
                x_mask_pad_right = np.zeros(
                    (chunk_size, max_len_for_x_mask_tmp), dtype=dtype)
                x_cur_pad = np.concatenate(
                    [x_mask_pad_left, x_mask_cur, x_mask_pad_right], axis=1)
                x_cur_pad = x_cur_pad[:chunk_size, :max_len_for_x_mask_tmp]
                x_add_mask_fsmn = np.concatenate([fsmn_padding, x_cur_pad],
                                                 axis=0)
                x_add_mask = np.concatenate([x_add_mask, x_add_mask_fsmn],
                                            axis=0)

                # x_mask rm
                fsmn_padding = np.zeros((max_len_for_x_mask_tmp, shfit_fsmn),
                                        dtype=dtype)
                x_mask_cur = np.diag(np.ones(stride, dtype=dtype))
                x_mask_right = np.zeros((stride, chunk_size - stride),
                                        dtype=dtype)
                x_mask_cur = np.concatenate([x_mask_cur, x_mask_right], axis=1)
                x_mask_cur_pad_top = np.zeros((chunk_ids * stride, chunk_size),
                                              dtype=dtype)
                x_mask_cur_pad_bottom = np.zeros(
                    (max_len_for_x_mask_tmp, chunk_size), dtype=dtype)
                x_rm_mask_cur = np.concatenate(
                    [x_mask_cur_pad_top, x_mask_cur, x_mask_cur_pad_bottom],
                    axis=0)
                x_rm_mask_cur = x_rm_mask_cur[:max_len_for_x_mask_tmp, :
                                              chunk_size]
                x_rm_mask_cur_fsmn = np.concatenate(
                    [fsmn_padding, x_rm_mask_cur], axis=1)
                x_rm_mask = np.concatenate([x_rm_mask, x_rm_mask_cur_fsmn],
                                           axis=1)

                # fsmn_padding_mask
                pad_shfit_mask = np.zeros([shfit_fsmn, num_units], dtype=dtype)
                ones_1 = np.ones([chunk_size, num_units], dtype=dtype)
                mask_shfit_chunk_cur = np.concatenate([pad_shfit_mask, ones_1],
                                                      axis=0)
                mask_shfit_chunk = np.concatenate(
                    [mask_shfit_chunk, mask_shfit_chunk_cur], axis=0)

                # predictor mask
                zeros_1 = np.zeros(
                    [shfit_fsmn + pad_left, num_units_predictor], dtype=dtype)
                ones_2 = np.ones([stride, num_units_predictor], dtype=dtype)
                zeros_3 = np.zeros(
                    [chunk_size - stride - pad_left, num_units_predictor],
                    dtype=dtype)
                ones_zeros = np.concatenate([ones_2, zeros_3], axis=0)
                mask_chunk_predictor_cur = np.concatenate(
                    [zeros_1, ones_zeros], axis=0)
                mask_chunk_predictor = np.concatenate(
                    [mask_chunk_predictor, mask_chunk_predictor_cur], axis=0)

                # encoder att mask
                zeros_1_top = np.zeros(
                    [shfit_fsmn, chunk_num * chunk_size_pad_shift],
                    dtype=dtype)

                zeros_2_num = max(chunk_ids - encoder_att_look_back_factor, 0)
                zeros_2 = np.zeros(
                    [chunk_size, zeros_2_num * chunk_size_pad_shift],
                    dtype=dtype)

                encoder_att_look_back_num = max(chunk_ids - zeros_2_num, 0)
                zeros_2_left = np.zeros([chunk_size, shfit_fsmn], dtype=dtype)
                ones_2_mid = np.ones([stride, stride], dtype=dtype)
                zeros_2_bottom = np.zeros([chunk_size - stride, stride],
                                          dtype=dtype)
                zeros_2_right = np.zeros([chunk_size, chunk_size - stride],
                                         dtype=dtype)
                ones_2 = np.concatenate([ones_2_mid, zeros_2_bottom], axis=0)
                ones_2 = np.concatenate([zeros_2_left, ones_2, zeros_2_right],
                                        axis=1)
                ones_2 = np.tile(ones_2, [1, encoder_att_look_back_num])

                zeros_3_left = np.zeros([chunk_size, shfit_fsmn], dtype=dtype)
                ones_3_right = np.ones([chunk_size, chunk_size], dtype=dtype)
                ones_3 = np.concatenate([zeros_3_left, ones_3_right], axis=1)

                zeros_remain_num = max(chunk_num - 1 - chunk_ids, 0)
                zeros_remain = np.zeros(
                    [chunk_size, zeros_remain_num * chunk_size_pad_shift],
                    dtype=dtype)

                ones2_bottom = np.concatenate(
                    [zeros_2, ones_2, ones_3, zeros_remain], axis=1)
                mask_att_chunk_encoder_cur = np.concatenate(
                    [zeros_1_top, ones2_bottom], axis=0)
                mask_att_chunk_encoder = np.concatenate(
                    [mask_att_chunk_encoder, mask_att_chunk_encoder_cur],
                    axis=0)

                # decoder fsmn_shift_att_mask
                zeros_1 = np.zeros([shfit_fsmn, 1])
                ones_1 = np.ones([chunk_size, 1])
                mask_shift_att_chunk_decoder_cur = np.concatenate(
                    [zeros_1, ones_1], axis=0)
                mask_shift_att_chunk_decoder = np.concatenate(
                    [
                        mask_shift_att_chunk_decoder,
                        mask_shift_att_chunk_decoder_cur
                    ],
                    vaxis=0)  # noqa: *

            self.x_add_mask = x_add_mask[:x_len_chunk_max, :x_len_max]
            self.x_len_chunk = x_len_chunk
            self.x_rm_mask = x_rm_mask[:x_len_max, :x_len_chunk_max]
            self.x_len = x_len
            self.mask_shfit_chunk = mask_shfit_chunk[:x_len_chunk_max, :]
            self.mask_chunk_predictor = mask_chunk_predictor[:
                                                             x_len_chunk_max, :]
            self.mask_att_chunk_encoder = mask_att_chunk_encoder[:
                                                                 x_len_chunk_max, :
                                                                 x_len_chunk_max]
            self.mask_shift_att_chunk_decoder = mask_shift_att_chunk_decoder[:
                                                                             x_len_chunk_max, :]

        return (self.x_add_mask, self.x_len_chunk, self.x_rm_mask, self.x_len,
                self.mask_shfit_chunk, self.mask_chunk_predictor,
                self.mask_att_chunk_encoder, self.mask_shift_att_chunk_decoder)

    def split_chunk(self, x, x_len, chunk_outs):
        """
        :param x: (b, t, d)
        :param x_length: (b)
        :param ind: int
        :return:
        """
        x = x[:, :x_len.max(), :]
        b, t, d = x.size()
        x_len_mask = (~make_pad_mask(x_len, maxlen=t)).to(x.device)
        x *= x_len_mask[:, :, None]

        x_add_mask = self.get_x_add_mask(chunk_outs, x.device, dtype=x.dtype)
        x_len_chunk = self.get_x_len_chunk(
            chunk_outs, x_len.device, dtype=x_len.dtype)
        x = torch.transpose(x, 1, 0)
        x = torch.reshape(x, [t, -1])
        x_chunk = torch.mm(x_add_mask, x)
        x_chunk = torch.reshape(x_chunk, [-1, b, d]).transpose(1, 0)

        return x_chunk, x_len_chunk

    def remove_chunk(self, x_chunk, x_len_chunk, chunk_outs):
        x_chunk = x_chunk[:, :x_len_chunk.max(), :]
        b, t, d = x_chunk.size()
        x_len_chunk_mask = (~make_pad_mask(x_len_chunk, maxlen=t)).to(
            x_chunk.device)
        x_chunk *= x_len_chunk_mask[:, :, None]

        x_rm_mask = self.get_x_rm_mask(
            chunk_outs, x_chunk.device, dtype=x_chunk.dtype)
        x_len = self.get_x_len(
            chunk_outs, x_len_chunk.device, dtype=x_len_chunk.dtype)
        x_chunk = torch.transpose(x_chunk, 1, 0)
        x_chunk = torch.reshape(x_chunk, [t, -1])
        x = torch.mm(x_rm_mask, x_chunk)
        x = torch.reshape(x, [-1, b, d]).transpose(1, 0)

        return x, x_len

    def get_x_add_mask(self, chunk_outs, device, idx=0, dtype=torch.float32):
        x = chunk_outs[idx]
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_x_len_chunk(self, chunk_outs, device, idx=1, dtype=torch.float32):
        x = chunk_outs[idx]
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_x_rm_mask(self, chunk_outs, device, idx=2, dtype=torch.float32):
        x = chunk_outs[idx]
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_x_len(self, chunk_outs, device, idx=3, dtype=torch.float32):
        x = chunk_outs[idx]
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_mask_shfit_chunk(self,
                             chunk_outs,
                             device,
                             batch_size=1,
                             num_units=1,
                             idx=4,
                             dtype=torch.float32):
        x = chunk_outs[idx]
        x = np.tile(x[None, :, :, ], [batch_size, 1, num_units])
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_mask_chunk_predictor(self,
                                 chunk_outs,
                                 device,
                                 batch_size=1,
                                 num_units=1,
                                 idx=5,
                                 dtype=torch.float32):
        x = chunk_outs[idx]
        x = np.tile(x[None, :, :, ], [batch_size, 1, num_units])
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_mask_att_chunk_encoder(self,
                                   chunk_outs,
                                   device,
                                   batch_size=1,
                                   idx=6,
                                   dtype=torch.float32):
        x = chunk_outs[idx]
        x = np.tile(x[None, :, :, ], [batch_size, 1, 1])
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()

    def get_mask_shift_att_chunk_decoder(self,
                                         chunk_outs,
                                         device,
                                         batch_size=1,
                                         idx=7,
                                         dtype=torch.float32):
        x = chunk_outs[idx]
        x = np.tile(x[None, None, :, 0], [batch_size, 1, 1])
        x = torch.from_numpy(x).type(dtype).to(device)
        return x.detach()
