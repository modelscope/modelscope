# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, dropatt=0.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropatt = nn.Dropout(dropatt)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropatt(attn)
        output = torch.bmm(attn, v)

        return output, attn


class Prenet(nn.Module):

    def __init__(self, in_units, prenet_units, out_units=0):
        super(Prenet, self).__init__()

        self.fcs = nn.ModuleList()
        for in_dim, out_dim in zip([in_units] + prenet_units[:-1],
                                   prenet_units):
            self.fcs.append(nn.Linear(in_dim, out_dim))
            self.fcs.append(nn.ReLU())
            self.fcs.append(nn.Dropout(0.5))

        if out_units:
            self.fcs.append(nn.Linear(prenet_units[-1], out_units))

    def forward(self, input):
        output = input
        for layer in self.fcs:
            output = layer(output)
        return output


class MultiHeadSelfAttention(nn.Module):
    """ Multi-Head SelfAttention module """

    def __init__(self, n_head, d_in, d_model, d_head, dropout, dropatt=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_in = d_in
        self.d_model = d_model

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.w_qkv = nn.Linear(d_in, 3 * n_head * d_head)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_head, 0.5), dropatt=dropatt)

        self.fc = nn.Linear(n_head * d_head, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        d_head, n_head = self.d_head, self.n_head

        sz_b, len_in, _ = input.size()

        residual = input

        x = self.layer_norm(input)
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, -1)

        q = q.view(sz_b, len_in, n_head, d_head)
        k = k.view(sz_b, len_in, n_head, d_head)
        v = v.view(sz_b, len_in, n_head, d_head)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_in,
                                                    d_head)  # (n*b) x l x d
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_in,
                                                    d_head)  # (n*b) x l x d
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_in,
                                                    d_head)  # (n*b) x l x d

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_in, d_head)
        output = (output.permute(1, 2, 0,
                                 3).contiguous().view(sz_b, len_in,
                                                      -1))  # b x l x (n*d)

        output = self.dropout(self.fc(output))
        if output.size(-1) == residual.size(-1):
            output = output + residual

        return output, attn


class PositionwiseConvFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self,
                 d_in,
                 d_hid,
                 kernel_size=(3, 1),
                 dropout_inner=0.1,
                 dropout=0.1):
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
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout_inner = nn.Dropout(dropout_inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm(x)

        output = x.transpose(1, 2)
        output = F.relu(self.w_1(output))
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(1), 0)
        output = self.dropout_inner(output)
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)

        output = output + residual

        return output


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_in,
        d_model,
        n_head,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropout_attn=0.0,
        dropout_relu=0.0,
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadSelfAttention(
            n_head,
            d_in,
            d_model,
            d_head,
            dropout=dropout,
            dropatt=dropout_attn)
        self.pos_ffn = PositionwiseConvFeedForward(
            d_model,
            d_inner,
            kernel_size,
            dropout_inner=dropout_relu,
            dropout=dropout)

    def forward(self, input, mask=None, slf_attn_mask=None):
        output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        output = self.pos_ffn(output, mask=mask)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn


class MultiHeadPNCAAttention(nn.Module):
    """ Multi-Head Attention PNCA module """

    def __init__(self, n_head, d_model, d_mem, d_head, dropout, dropatt=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.d_mem = d_mem

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.w_x_qkv = nn.Linear(d_model, 3 * n_head * d_head)
        self.fc_x = nn.Linear(n_head * d_head, d_model)

        self.w_h_kv = nn.Linear(d_mem, 2 * n_head * d_head)
        self.fc_h = nn.Linear(n_head * d_head, d_model)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_head, 0.5), dropatt=dropatt)

        self.dropout = nn.Dropout(dropout)

    def update_x_state(self, x):
        d_head, n_head = self.d_head, self.n_head

        sz_b, len_x, _ = x.size()

        x_qkv = self.w_x_qkv(x)
        x_q, x_k, x_v = x_qkv.chunk(3, -1)

        x_q = x_q.view(sz_b, len_x, n_head, d_head)
        x_k = x_k.view(sz_b, len_x, n_head, d_head)
        x_v = x_v.view(sz_b, len_x, n_head, d_head)

        x_q = x_q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
        x_k = x_k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
        x_v = x_v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)

        if self.x_state_size:
            self.x_k = torch.cat([self.x_k, x_k], dim=1)
            self.x_v = torch.cat([self.x_v, x_v], dim=1)
        else:
            self.x_k = x_k
            self.x_v = x_v

        self.x_state_size += len_x

        return x_q, x_k, x_v

    def update_h_state(self, h):
        if self.h_state_size == h.size(1):
            return None, None

        d_head, n_head = self.d_head, self.n_head

        # H
        sz_b, len_h, _ = h.size()

        h_kv = self.w_h_kv(h)
        h_k, h_v = h_kv.chunk(2, -1)

        h_k = h_k.view(sz_b, len_h, n_head, d_head)
        h_v = h_v.view(sz_b, len_h, n_head, d_head)

        self.h_k = h_k.permute(2, 0, 1, 3).contiguous().view(-1, len_h, d_head)
        self.h_v = h_v.permute(2, 0, 1, 3).contiguous().view(-1, len_h, d_head)

        self.h_state_size += len_h

        return h_k, h_v

    def reset_state(self):
        self.h_k = None
        self.h_v = None
        self.h_state_size = 0
        self.x_k = None
        self.x_v = None
        self.x_state_size = 0

    def forward(self, x, h, mask_x=None, mask_h=None):
        residual = x
        self.update_h_state(h)
        x_q, x_k, x_v = self.update_x_state(self.layer_norm(x))

        d_head, n_head = self.d_head, self.n_head

        sz_b, len_in, _ = x.size()

        # X
        if mask_x is not None:
            mask_x = mask_x.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output_x, attn_x = self.attention(x_q, self.x_k, self.x_v, mask=mask_x)

        output_x = output_x.view(n_head, sz_b, len_in, d_head)
        output_x = (output_x.permute(1, 2, 0,
                                     3).contiguous().view(sz_b, len_in,
                                                          -1))  # b x l x (n*d)
        output_x = self.fc_x(output_x)

        # H
        if mask_h is not None:
            mask_h = mask_h.repeat(n_head, 1, 1)
        output_h, attn_h = self.attention(x_q, self.h_k, self.h_v, mask=mask_h)

        output_h = output_h.view(n_head, sz_b, len_in, d_head)
        output_h = (output_h.permute(1, 2, 0,
                                     3).contiguous().view(sz_b, len_in,
                                                          -1))  # b x l x (n*d)
        output_h = self.fc_h(output_h)

        output = output_x + output_h

        output = self.dropout(output)

        output = output + residual

        return output, attn_x, attn_h


class PNCABlock(nn.Module):
    """PNCA Block"""

    def __init__(
        self,
        d_model,
        d_mem,
        n_head,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropout_attn=0.0,
        dropout_relu=0.0,
    ):
        super(PNCABlock, self).__init__()
        self.pnca_attn = MultiHeadPNCAAttention(
            n_head,
            d_model,
            d_mem,
            d_head,
            dropout=dropout,
            dropatt=dropout_attn)
        self.pos_ffn = PositionwiseConvFeedForward(
            d_model,
            d_inner,
            kernel_size,
            dropout_inner=dropout_relu,
            dropout=dropout)

    def forward(self,
                input,
                memory,
                mask=None,
                pnca_x_attn_mask=None,
                pnca_h_attn_mask=None):
        output, pnca_attn_x, pnca_attn_h = self.pnca_attn(
            input, memory, pnca_x_attn_mask, pnca_h_attn_mask)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        output = self.pos_ffn(output, mask=mask)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, pnca_attn_x, pnca_attn_h

    def reset_state(self):
        self.pnca_attn.reset_state()
