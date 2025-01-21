# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_def import (HEADER_BLOCK_SIZE, ActivationType, LayerType, f32ToI32,
                        printNeonMatrix, printNeonVector)

DEBUG = False


def to_kaldi_matrix(np_mat):
    """ function that transform as str numpy mat to standard kaldi str matrix

        Args:
            np_mat:          numpy mat

        Returns:  str
    """
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


def print_tensor(torch_tensor):
    """ print torch tensor for debug

    Args:
        torch_tensor:           a tensor
    """
    re_str = ''
    x = torch_tensor.detach().squeeze().numpy()
    re_str += to_kaldi_matrix(x)
    re_str += '<!EndOfComponent>\n'
    print(re_str)


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        self.debug = False
        self.dataout = None

    def forward(self, input):
        output = self.linear(input)

        if self.debug:
            self.dataout = output

        return output

    def print_model(self):
        printNeonMatrix(self.linear.weight)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<LinearTransform> %d %d\n' % (self.output_dim,
                                                 self.input_dim)
        re_str += '<LearnRateCoef> 1\n'

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        re_str += '<!EndOfComponent>\n'

        return re_str


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        self.debug = False
        self.dataout = None

    def forward(self, input):
        output = self.linear(input)

        if self.debug:
            self.dataout = output

        return output

    def print_model(self):
        printNeonMatrix(self.linear.weight)
        printNeonVector(self.linear.bias)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<AffineTransform> %d %d\n' % (self.output_dim,
                                                 self.input_dim)
        re_str += '<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0\n'

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)

        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        re_str += '<!EndOfComponent>\n'

        return re_str


class Fsmn(nn.Module):
    """
    FSMN implementation.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 lorder=None,
                 rorder=None,
                 lstride=None,
                 rstride=None):
        super(Fsmn, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim,
            self.dim, (lorder, 1),
            dilation=(lstride, 1),
            groups=self.dim,
            bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim,
                self.dim, (rorder, 1),
                dilation=(rstride, 1),
                groups=self.dim,
                bias=False)
        else:
            self.conv_right = None

        self.debug = False
        self.dataout = None

    def forward(self, input):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)

        out1 = out.permute(0, 3, 2, 1)
        output = out1.squeeze(1)

        if self.debug:
            self.dataout = output

        return output

    def print_model(self):
        tmpw = self.conv_left.weight
        tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
        for j in range(tmpw.shape[0]):
            tmpwm[:, j] = tmpw[j, 0, :, 0]

        printNeonMatrix(tmpwm)

        if self.conv_right is not None:
            tmpw = self.conv_right.weight
            tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
            for j in range(tmpw.shape[0]):
                tmpwm[:, j] = tmpw[j, 0, :, 0]

            printNeonMatrix(tmpwm)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<Fsmn> %d %d\n' % (self.dim, self.dim)
        re_str += '<LearnRateCoef> %d <LOrder> %d <ROrder> %d <LStride> %d <RStride> %d <MaxNorm> 0\n' % (
            1, self.lorder, self.rorder, self.lstride, self.rstride)

        lfiters = self.state_dict()['conv_left.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += to_kaldi_matrix(x)

        if self.conv_right is not None:
            rfiters = self.state_dict()['conv_right.weight']
            x = (rfiters.squeeze().numpy().T)
            re_str += to_kaldi_matrix(x)
            re_str += '<!EndOfComponent>\n'

        return re_str


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(input)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<RectifiedLinear> %d %d\n' % (self.dim, self.dim)
        re_str += '<!EndOfComponent>\n'
        return re_str


class FSMNNet(nn.Module):
    """
    FSMN net for keyword spotting
    """

    def __init__(self,
                 input_dim=200,
                 linear_dim=128,
                 proj_dim=128,
                 lorder=10,
                 rorder=1,
                 num_syn=5,
                 fsmn_layers=4):
        """
        Args:
            input_dim:              input dimension
            linear_dim:             fsmn input dimension
            proj_dim:               fsmn projection dimension
            lorder:                 fsmn left order
            rorder:                 fsmn right order
            num_syn:                output dimension
            fsmn_layers:            no. of sequential fsmn layers
        """
        super(FSMNNet, self).__init__()

        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.rorder = rorder
        self.num_syn = num_syn
        self.fsmn_layers = fsmn_layers

        self.linear1 = AffineTransform(input_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

        self.fsmn = self._build_repeats(linear_dim, proj_dim, lorder, rorder,
                                        fsmn_layers)

        self.linear2 = AffineTransform(linear_dim, num_syn)

    @staticmethod
    def _build_repeats(linear_dim=136,
                       proj_dim=68,
                       lorder=3,
                       rorder=2,
                       fsmn_layers=5):
        repeats = [
            nn.Sequential(
                LinearTransform(linear_dim, proj_dim),
                Fsmn(proj_dim, proj_dim, lorder, rorder, 1, 1),
                AffineTransform(proj_dim, linear_dim),
                RectifiedLinear(linear_dim, linear_dim))
            for i in range(fsmn_layers)
        ]

        return nn.Sequential(*repeats)

    def forward(self, input):
        x1 = self.linear1(input)
        x2 = self.relu(x1)
        x3 = self.fsmn(x2)
        x4 = self.linear2(x3)
        return x4

    def print_model(self):
        self.linear1.print_model()

        for layer in self.fsmn:
            layer[0].print_model()
            layer[1].print_model()
            layer[2].print_model()

        self.linear2.print_model()

    def print_header(self):
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 0.0
        # dimins
        header[2] = self.input_dim
        # dimouts
        header[3] = self.num_syn
        # numlayers
        header[4] = 3

        #
        # write each layer's header
        #
        hidx = 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(
            ActivationType.ACTIVATION_RELU.value)
        hidx += 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = self.lorder
        header[HEADER_BLOCK_SIZE * hidx + 5] = self.rorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = self.fsmn_layers
        header[HEADER_BLOCK_SIZE * hidx + 7] = -1.0
        hidx += 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(
            ActivationType.ACTIVATION_SOFTMAX.value)

        for h in header:
            print(f32ToI32(h))

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.linear1.to_kaldi_nnet()
        re_str += self.relu.to_kaldi_nnet()

        for fsmn in self.fsmn:
            re_str += fsmn[0].to_kaldi_nnet()
            re_str += fsmn[1].to_kaldi_nnet()
            re_str += fsmn[2].to_kaldi_nnet()
            re_str += fsmn[3].to_kaldi_nnet()

        re_str += self.linear2.to_kaldi_nnet()
        re_str += '<Softmax> %d %d\n' % (self.num_syn, self.num_syn)
        re_str += '<!EndOfComponent>\n'
        re_str += '</Nnet>\n'

        return re_str


class DFSMN(nn.Module):
    """
    One deep fsmn layer
    """

    def __init__(self,
                 dimproj=64,
                 dimlinear=128,
                 lorder=20,
                 rorder=1,
                 lstride=1,
                 rstride=1):
        """
        Args:
            dimproj:                projection dimension, input and output dimension of memory blocks
            dimlinear:              dimension of mapping layer
            lorder:                 left order
            rorder:                 right order
            lstride:                left stride
            rstride:                right stride
        """
        super(DFSMN, self).__init__()

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.expand = AffineTransform(dimproj, dimlinear)
        self.shrink = LinearTransform(dimlinear, dimproj)

        self.conv_left = nn.Conv2d(
            dimproj,
            dimproj, (lorder, 1),
            dilation=(lstride, 1),
            groups=dimproj,
            bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                dimproj,
                dimproj, (rorder, 1),
                dilation=(rstride, 1),
                groups=dimproj,
                bias=False)
        else:
            self.conv_right = None

    def forward(self, input):
        f1 = F.relu(self.expand(input))
        p1 = self.shrink(f1)

        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)

        out1 = out.permute(0, 3, 2, 1)
        output = input + out1.squeeze(1)

        return output

    def print_model(self):
        self.expand.print_model()
        self.shrink.print_model()

        tmpw = self.conv_left.weight
        tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
        for j in range(tmpw.shape[0]):
            tmpwm[:, j] = tmpw[j, 0, :, 0]

        printNeonMatrix(tmpwm)

        if self.conv_right is not None:
            tmpw = self.conv_right.weight
            tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
            for j in range(tmpw.shape[0]):
                tmpwm[:, j] = tmpw[j, 0, :, 0]

            printNeonMatrix(tmpwm)


def build_dfsmn_repeats(linear_dim=128,
                        proj_dim=64,
                        lorder=20,
                        rorder=1,
                        fsmn_layers=6):
    """
    build stacked dfsmn layers
    Args:
        linear_dim:
        proj_dim:
        lorder:
        rorder:
        fsmn_layers:

    Returns:

    """
    repeats = [
        nn.Sequential(DFSMN(proj_dim, linear_dim, lorder, rorder, 1, 1))
        for i in range(fsmn_layers)
    ]

    return nn.Sequential(*repeats)
