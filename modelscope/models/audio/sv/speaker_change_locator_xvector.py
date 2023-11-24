# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.TDNN import Xvector
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, n_units, h=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.att = None

    def forward(self, x, batch_size):
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.transpose(1, 2), k.permute(
            0, 2, 3, 1)) / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        # v : (B, T, h, d_k)
        # p_att : (B, h, T, T)
        x = torch.matmul(p_att, v.transpose(1, 2))
        # x : (B, h, T, d_k)
        x = x.transpose(1, 2).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, n_units, d_units, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(n_units, d_units)
        self.linear2 = nn.Linear(d_units, n_units)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PosEncoding(nn.Module):

    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array([[
            pos / np.power(10000, 2.0 * (j // 2) / d_word_vec)
            for j in range(d_word_vec)
        ] for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        self.pos_enc = torch.nn.Embedding(max_seq_len + 1, d_word_vec)
        self.pos_enc.weight = torch.nn.Parameter(
            torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        input_pos = torch.LongTensor([
            list(range(1, len + 1)) + [0] * (max_len - len)
            for len in input_len
        ])

        input_pos = input_pos.to(list(self.pos_enc.parameters())[0].device)
        return self.pos_enc(input_pos)


class TransformerEncoder(nn.Module):

    def __init__(self,
                 idim,
                 n_units=256,
                 n_layers=2,
                 e_units=512,
                 h=4,
                 dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format('lnorm1_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('self_att_', i),
                    MultiHeadSelfAttention(n_units, h))
            setattr(self, '{}{:d}'.format('lnorm2_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('ff_', i),
                    PositionwiseFeedForward(n_units, e_units, dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        # x: [B, num_anchors, T, n_in]
        bs, num, tframe, dim = x.size()
        x = x.reshape(bs * num, tframe, -1)  # [B*num_anchors, T, dim]
        # x: (B, T, F) ... batch, time, (mel)freq
        B_size, T_size, _ = x.shape
        # e: (BT, F)
        e = self.linear_in(x.reshape(B_size * T_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format('lnorm1_', i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format('self_att_', i))(e, x.shape[0])
            # residual
            e = e + self.dropout(s)
            # layer normalization
            e = getattr(self, '{}{:d}'.format('lnorm2_', i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format('ff_', i))(e)
            # residual
            e = e + self.dropout(s)
        # final layer normalization
        # output: (BT, F)
        # output: (B, F, T)
        output = self.lnorm_out(e).reshape(B_size, T_size, -1)
        output = output.reshape(bs, num, tframe,
                                -1)  # [B, num_anchors, T, dim]
        return output


class TransformerEncoder_out(nn.Module):

    def __init__(self,
                 idim,
                 n_units=256,
                 n_layers=2,
                 e_units=512,
                 h=4,
                 dropout=0.1):
        super(TransformerEncoder_out, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)

        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format('lnorm1_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('self_att_', i),
                    MultiHeadSelfAttention(n_units, h))
            setattr(self, '{}{:d}'.format('lnorm2_', i), nn.LayerNorm(n_units))
            setattr(self, '{}{:d}'.format('ff_', i),
                    PositionwiseFeedForward(n_units, e_units, dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        # x: (B, T, F)
        B_size, T_size, _ = x.shape
        # e: (BT, F)
        e = self.linear_in(x.reshape(B_size * T_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format('lnorm1_', i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format('self_att_', i))(e, x.shape[0])
            # residual
            e = e + self.dropout(s)
            # layer normalization
            e = getattr(self, '{}{:d}'.format('lnorm2_', i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format('ff_', i))(e)
            # residual
            e = e + self.dropout(s)
        # final layer normalization
        # output: (BT, F)
        # output: (B, T, F)
        output = self.lnorm_out(e).reshape(B_size, T_size, -1)
        return output


class OutLayer(nn.Module):

    def __init__(self, n_units=256, num_anchors=2):
        super(OutLayer, self).__init__()
        self.rnn_combine = TransformerEncoder_out(num_anchors * n_units,
                                                  n_units)
        self.out_linear = nn.Linear(n_units // num_anchors, 1)

    def forward(self, input):
        # input: [B, num_anchors, T, dim]
        bs, num, tframe, dim = input.size()
        output = input.permute(0, 2, 1,
                               3).reshape(bs, tframe,
                                          -1)  # [Bs, t, num_anchors*dim]
        output = self.rnn_combine(output)  # [Bs, t, n_units]
        output = output.reshape(
            bs, tframe, num, -1)  # [Bs, t, num_anchors, n_units//num_anchors]
        output = self.out_linear(output).squeeze(-1)  # [Bs, t, num_anchors]

        return output


class TransformerDetector(nn.Module):

    def __init__(self,
                 frame_dim=512,
                 anchor_dim=192,
                 hidden_dim=256,
                 max_seq_len=500):
        super(TransformerDetector, self).__init__()
        self.detection = TransformerEncoder(
            idim=frame_dim + anchor_dim, n_units=hidden_dim)
        self.output = OutLayer(n_units=hidden_dim)
        self.pos_enc = PosEncoding(max_seq_len, hidden_dim)

    def forward(self, feats, anchors):
        # feats: [1, t, fdim]
        num_frames = feats.shape[1]
        num_anchors = anchors.shape[1]
        bs = feats.shape[0]
        feats = feats.unsqueeze(1).repeat(
            1, num_anchors, 1, 1)  # shape: [Bs, num_anchors, t, fdim]
        anchors = anchors.unsqueeze(2).repeat(
            1, 1, num_frames, 1)  # shape: [Bs, num_anchors, t, xdim]
        sd_in = torch.cat((feats, anchors),
                          dim=-1)  # shape: [Bs, num_anchors, t, fdim+xdim]
        sd_out = self.detection(sd_in)  # shape: [Bs, num_anchors, t, sd_dim]

        # pos
        pos_emb = self.pos_enc(torch.tensor([num_frames] * (bs * num_anchors)))
        pos_emb = pos_emb.reshape(bs, num_anchors, num_frames, -1)
        sd_out += pos_emb

        # output
        output = self.output(sd_out)  # shape: [Bs, t, num_anchors]

        return output


@MODELS.register_module(
    Tasks.speaker_diarization, module_name=Models.scl_sd_xvector)
class SpeakerChangeLocatorTransformer(TorchModel):
    r"""A speaekr change locator using the transformer architecture as the backbone.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config

        self.feature_dim = self.model_config['fbank_dim']
        frame_size = self.model_config['frame_size']
        anchor_size = self.model_config['anchor_size']
        self.device = create_device(kwargs['device'])

        self.encoder = Xvector(in_channels=self.feature_dim)
        self.backend = TransformerDetector(
            frame_dim=frame_size, anchor_dim=anchor_size)

        pretrained_encoder = kwargs['pretrained_encoder']
        pretrained_backend = kwargs['pretrained_backend']

        self.__load_check_point(pretrained_encoder, pretrained_backend)

        self.encoder.to(self.device)
        self.backend.to(self.device)
        self.encoder.eval()
        self.backend.eval()

    def forward(self, audio, anchors):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        assert len(audio.shape) == 2 and audio.shape[
            0] == 1, 'modelscope error: the shape of input audio to model needs to be [1, T]'
        assert len(
            anchors.shape
        ) == 3 and anchors.shape[0] == 1 and anchors.shape[
            1] == 2, 'modelscope error: the shape of input anchors to model needs to be [1, 2, D]'
        # audio shape: [1, T]
        feature = self.__extract_feature(audio)
        frame_state = self.encoder(feature.to(self.device))
        output = self.backend(frame_state, anchors.to(self.device))
        output = output.squeeze(0).detach().cpu().sigmoid()

        time_scale_factor = int(np.ceil(feature.shape[1] / output.shape[0]))
        output = output.unsqueeze(1).expand(-1, time_scale_factor,
                                            -1).reshape(-1, output.shape[-1])
        return output

    def __extract_feature(self, audio):
        feature = Kaldi.fbank(audio, num_mel_bins=self.feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        return feature

    def __load_check_point(
        self,
        pretrained_encoder,
        pretrained_backend,
    ):
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_encoder),
                map_location=torch.device('cpu')))

        self.backend.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_backend),
                map_location=torch.device('cpu')))
