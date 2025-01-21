# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .layer_base import (LayerBase, expect_kaldi_matrix, expect_token_number,
                         to_kaldi_matrix)


class DeepFsmn(LayerBase):

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=None,
                 rorder=None,
                 hidden_size=None,
                 layer_norm=False,
                 dropout=0):
        super(DeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm

        self.linear = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim, [lorder, 1], [1, 1],
            groups=output_dim,
            bias=False)
        self.conv2 = nn.Conv2d(
            output_dim,
            output_dim, [rorder, 1], [1, 1],
            groups=output_dim,
            bias=False)

    def forward(self, input):

        f1 = F.relu(self.linear(input))

        f1 = self.drop1(f1)
        if self.layer_norm:
            f1 = self.norm(f1)

        p1 = self.project(f1)

        x = th.unsqueeze(p1, 1)

        x_per = x.permute(0, 3, 2, 1)

        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])
        yr = F.pad(x_per, [0, 0, 0, self.rorder])
        yr = yr[:, :, 1:, :]

        out = x_per + self.conv1(y) + self.conv2(yr)
        out = self.drop2(out)

        out1 = out.permute(0, 3, 2, 1)

        return input + out1.squeeze()

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<UniDeepFsmn> %d %d\n'\
                  % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> %d <HidSize> %d <LOrder> %d <LStride> %d <MaxNorm> 0\n'\
                  % (1, self.hidden_size, self.lorder, 1)
        lfiters = self.state_dict()['conv1.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += to_kaldi_matrix(x)
        proj_weights = self.state_dict()['project.weight']
        x = proj_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        return re_str

    def load_kaldi_nnet(self, instr):
        output = expect_token_number(
            instr,
            '<LearnRateCoef>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error for <LearnRateCoef>')
        instr, lr = output

        output = expect_token_number(
            instr,
            '<HidSize>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error for <HidSize>')
        instr, hiddensize = output
        self.hidden_size = int(hiddensize)

        output = expect_token_number(
            instr,
            '<LOrder>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error for <LOrder>')
        instr, lorder = output
        self.lorder = int(lorder)

        output = expect_token_number(
            instr,
            '<LStride>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error for <LStride>')
        instr, lstride = output
        self.lstride = lstride

        output = expect_token_number(
            instr,
            '<MaxNorm>',
        )
        if output is None:
            raise Exception('UniDeepFsmn format error for <MaxNorm>')

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error for parsing matrix')
        instr, mat = output
        mat1 = np.fliplr(mat.T).copy()
        self.conv1 = nn.Conv2d(
            self.output_dim,
            self.output_dim, [self.lorder, 1], [1, 1],
            groups=self.output_dim,
            bias=False)
        mat_th = th.from_numpy(mat1).type(th.FloatTensor)
        mat_th = mat_th.unsqueeze(1)
        mat_th = mat_th.unsqueeze(3)
        self.conv1.weight = th.nn.Parameter(mat_th)

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error for parsing matrix')
        instr, mat = output

        self.project = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.linear = nn.Linear(self.input_dim, self.hidden_size)

        self.project.weight = th.nn.Parameter(
            th.from_numpy(mat).type(th.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error for parsing matrix')
        instr, mat = output
        self.linear.weight = th.nn.Parameter(
            th.from_numpy(mat).type(th.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('UniDeepFsmn format error for parsing matrix')
        instr, mat = output
        self.linear.bias = th.nn.Parameter(
            th.from_numpy(mat).type(th.FloatTensor))

        return instr
