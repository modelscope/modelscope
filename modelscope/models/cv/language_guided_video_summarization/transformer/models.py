# Part of the implementation is borrowed and modified from attention-is-all-you-need-pytorch,
# publicly available at https://github.com/jadore801120/attention-is-all-you-need-pytorch

import numpy as np
import torch
import torch.nn as nn

from .layers import DecoderLayer, EncoderLayer
from .sub_layers import MultiHeadAttention


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(self,
                 d_word_vec=1024,
                 n_layers=6,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=2048,
                 dropout=0.1,
                 n_position=200):

        super().__init__()

        self.position_enc = PositionalEncoding(
            d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, enc_output, return_attns=False):

        enc_slf_attn_list = []
        # -- Forward
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    """A decoder model with self attention mechanism."""

    def __init__(self,
                 d_word_vec=1024,
                 n_layers=6,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=2048,
                 n_position=200,
                 dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(
            d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self,
                dec_output,
                enc_output,
                src_mask=None,
                trg_mask=None,
                return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(self,
                 num_sentence=7,
                 txt_atten_head=4,
                 d_frame_vec=512,
                 d_model=512,
                 d_inner=2048,
                 n_layers=6,
                 n_head=8,
                 d_k=256,
                 d_v=256,
                 dropout=0.1,
                 n_position=4000):

        super().__init__()

        self.d_model = d_model

        self.layer_norm_img_src = nn.LayerNorm(d_frame_vec, eps=1e-6)
        self.layer_norm_img_trg = nn.LayerNorm(d_frame_vec, eps=1e-6)
        self.layer_norm_txt = nn.LayerNorm(
            num_sentence * d_frame_vec, eps=1e-6)

        self.linear_txt = nn.Linear(
            in_features=num_sentence * d_frame_vec, out_features=d_model)
        self.lg_attention = MultiHeadAttention(
            n_head=txt_atten_head, d_model=d_model, d_k=d_k, d_v=d_v)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_frame_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_frame_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_frame_vec, 'the dimensions of all module outputs shall be the same.'

        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_2 = nn.Linear(
            in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.norm_linear = nn.LayerNorm(
            normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_seq, src_txt, trg_seq):

        features_txt = self.linear_txt(src_txt)
        atten_seq, txt_attn = self.lg_attention(src_seq, features_txt,
                                                features_txt)

        enc_output, *_ = self.encoder(atten_seq)
        dec_output, *_ = self.decoder(trg_seq, enc_output)

        y = self.drop(enc_output)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, dec_output
