# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .layer_base import (LayerBase, expect_kaldi_matrix, expect_token_number,
                         to_kaldi_matrix)


class SepConv(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 out_channels,
                 kernel_size=(5, 2),
                 dilation=(1, 1)):
        """ :param kernel_size (time, frequency)

        """
        super(SepConv, self).__init__()
        # depthwise + pointwise
        self.dconv = nn.Conv2d(
            in_channels,
            in_channels * filters,
            kernel_size,
            dilation=dilation,
            groups=in_channels)
        self.pconv = nn.Conv2d(
            in_channels * filters, out_channels, kernel_size=1)
        self.padding = dilation[0] * (kernel_size[0] - 1)

    def forward(self, input):
        ''' input: [B, C, T, F]
        '''
        x = F.pad(input, [0, 0, self.padding, 0])
        x = self.dconv(x)
        x = self.pconv(x)
        return x


class Conv2d(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=20,
                 rorder=0,
                 groups=1,
                 bias=False,
                 skip_connect=True):
        super(Conv2d, self).__init__()
        self.lorder = lorder
        self.conv = nn.Conv2d(
            input_dim, output_dim, [lorder, 1], groups=groups, bias=bias)
        self.rorder = rorder
        if self.rorder:
            self.conv2 = nn.Conv2d(
                input_dim, output_dim, [rorder, 1], groups=groups, bias=bias)
        self.skip_connect = skip_connect

    def forward(self, input):
        # [B, 1, T, F]
        x = th.unsqueeze(input, 1)
        # [B, F, T, 1]
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])
        out = self.conv(y)
        if self.rorder:
            yr = F.pad(x_per, [0, 0, 0, self.rorder])
            yr = yr[:, :, 1:, :]
            out += self.conv2(yr)
        out = out.permute(0, 3, 2, 1).squeeze(1)
        if self.skip_connect:
            out = out + input
        return out


class SelfAttLayer(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(SelfAttLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.att = nn.Linear(input_dim, lorder, bias=False)

    def forward(self, input):

        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = th.unsqueeze(p1, 1)

        x_per = x.permute(0, 3, 2, 1)

        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        # z [B, F, T, lorder]
        z = x_per
        for i in range(1, self.lorder):
            z = th.cat([z, y[:, :, self.lorder - 1 - i:-i, :]], axis=-1)

        # [B, T, lorder]
        att = F.softmax(self.att(input), dim=-1)
        att = th.unsqueeze(att, 1)
        z = th.sum(z * att, axis=-1)

        out1 = z.permute(0, 2, 1)

        return input + out1


class TFFsmn(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=None,
                 hidden_size=None,
                 dilation=1,
                 layer_norm=False,
                 dropout=0,
                 skip_connect=True):
        super(TFFsmn, self).__init__()

        self.skip_connect = skip_connect

        self.linear = nn.Linear(input_dim, hidden_size)
        self.norm = nn.Identity()
        if layer_norm:
            self.norm = nn.LayerNorm(input_dim)
        self.act = nn.ReLU()
        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim, [lorder, 1],
            dilation=[dilation, 1],
            groups=output_dim,
            bias=False)
        self.padding_left = dilation * (lorder - 1)
        dorder = 5
        self.conv2 = nn.Conv2d(1, 1, [dorder, 1], bias=False)
        self.padding_freq = dorder - 1

    def forward(self, input):
        return self.compute1(input)

    def compute1(self, input):
        ''' linear-dconv-relu(norm)-linear-dconv
        '''
        x = self.linear(input)
        # [B, 1, F, T]
        x = th.unsqueeze(x, 1).permute(0, 1, 3, 2)
        z = F.pad(x, [0, 0, self.padding_freq, 0])
        z = self.conv2(z) + x
        x = z.permute(0, 3, 2, 1).squeeze(-1)
        x = self.act(x)
        x = self.norm(x)
        x = self.project(x)
        x = th.unsqueeze(x, 1).permute(0, 3, 2, 1)
        # [B, F, T+lorder-1, 1]
        y = F.pad(x, [0, 0, self.padding_left, 0])
        out = self.conv1(y)
        if self.skip_connect:
            out = out + x
        out = out.permute(0, 3, 2, 1).squeeze()

        return input + out


class CNNFsmn(nn.Module):
    ''' use cnn to reduce parameters
    '''

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=None,
                 hidden_size=None,
                 dilation=1,
                 layer_norm=False,
                 dropout=0,
                 skip_connect=True):
        super(CNNFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connect = skip_connect

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.act = nn.ReLU()
        kernel_size = (3, 8)
        stride = (1, 4)
        self.conv = nn.Sequential(
            nn.ConstantPad2d((stride[1], 0, kernel_size[0] - 1, 0), 0),
            nn.Conv2d(1, stride[1], kernel_size=kernel_size, stride=stride))

        self.dconv = nn.Conv2d(
            output_dim,
            output_dim, [lorder, 1],
            dilation=[dilation, 1],
            groups=output_dim,
            bias=False)
        self.padding_left = dilation * (lorder - 1)

    def forward(self, input):
        return self.compute2(input)

    def compute1(self, input):
        ''' linear-relu(norm)-conv2d-relu?-dconv
        '''
        # [B, T, F]
        x = self.linear(input)
        x = self.act(x)
        x = th.unsqueeze(x, 1)
        x = self.conv(x)
        # [B, C, T, F] -> [B, 1, T, F]
        b, c, t, f = x.shape
        x = x.view([b, 1, t, -1])
        x = x.permute(0, 3, 2, 1)
        # [B, F, T+lorder-1, 1]
        y = F.pad(x, [0, 0, self.padding_left, 0])
        out = self.dconv(y)
        if self.skip_connect:
            out = out + x
        out = out.permute(0, 3, 2, 1).squeeze()
        return input + out

    def compute2(self, input):
        ''' conv2d-relu-linear-relu?-dconv
        '''
        x = th.unsqueeze(input, 1)
        x = self.conv(x)
        x = self.act(x)
        # [B, C, T, F] -> [B, T, F]
        b, c, t, f = x.shape
        x = x.view([b, t, -1])
        x = self.linear(x)
        x = th.unsqueeze(x, 1).permute(0, 3, 2, 1)
        y = F.pad(x, [0, 0, self.padding_left, 0])
        out = self.dconv(y)
        if self.skip_connect:
            out = out + x
        out = out.permute(0, 3, 2, 1).squeeze()
        return input + out


class UniDeepFsmn(LayerBase):

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=None,
                 hidden_size=None,
                 dilation=1,
                 layer_norm=False,
                 dropout=0,
                 skip_connect=True):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connect = skip_connect

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)
        self.norm = nn.Identity()
        if layer_norm:
            self.norm = nn.LayerNorm(input_dim)
        self.act = nn.ReLU()
        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim, [lorder, 1],
            dilation=[dilation, 1],
            groups=output_dim,
            bias=False)
        self.padding_left = dilation * (lorder - 1)

    def forward(self, input):
        return self.compute1(input)

    def compute1(self, input):
        ''' linear-relu(norm)-linear-dconv
        '''
        # [B, T, F]
        x = self.linear(input)
        x = self.act(x)
        x = self.norm(x)
        x = self.project(x)
        x = th.unsqueeze(x, 1).permute(0, 3, 2, 1)
        # [B, F, T+lorder-1, 1]
        y = F.pad(x, [0, 0, self.padding_left, 0])
        out = self.conv1(y)
        if self.skip_connect:
            out = out + x
        out = out.permute(0, 3, 2, 1).squeeze()

        return input + out

    def compute2(self, input):
        ''' linear-dconv-linear-relu(norm)
        '''
        x = self.project(input)
        x = th.unsqueeze(x, 1).permute(0, 3, 2, 1)
        y = F.pad(x, [0, 0, self.padding_left, 0])
        out = self.conv1(y)
        if self.skip_connect:
            out = out + x
        out = out.permute(0, 3, 2, 1).squeeze()
        x = self.linear(out)
        x = self.act(x)
        x = self.norm(x)

        return input + x

    def compute3(self, input):
        ''' dconv-linear-relu(norm)-linear
        '''
        x = th.unsqueeze(input, 1).permute(0, 3, 2, 1)
        y = F.pad(x, [0, 0, self.padding_left, 0])
        out = self.conv1(y)
        if self.skip_connect:
            out = out + x
        out = out.permute(0, 3, 2, 1).squeeze()
        x = self.linear(out)
        x = self.act(x)
        x = self.norm(x)
        x = self.project(x)

        return input + x

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<UniDeepFsmn> %d %d\n' \
                  % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> %d <HidSize> %d <LOrder> %d <LStride> %d <MaxNorm> 0\n' \
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

    def to_raw_nnet(self, fid):
        lfiters = self.state_dict()['conv1.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        x.tofile(fid)

        proj_weights = self.state_dict()['project.weight']
        x = proj_weights.squeeze().numpy()
        x.tofile(fid)

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        x.tofile(fid)

        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        x.tofile(fid)

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
        mat = np.squeeze(mat)
        self.linear.bias = th.nn.Parameter(
            th.from_numpy(mat).type(th.FloatTensor))

        return instr
