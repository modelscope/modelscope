# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsmn import AffineTransform, Fsmn, LinearTransform, RectifiedLinear
from .model_def import HEADER_BLOCK_SIZE, ActivationType, LayerType, f32ToI32


class FSMNUnit(nn.Module):
    """ A multi-channel fsmn unit

    """

    def __init__(self, dimlinear=128, dimproj=64, lorder=20, rorder=1):
        """
        Args:
            dimlinear:              input / output dimension
            dimproj:                fsmn input / output dimension
            lorder:                 left order
            rorder:                 right order
        """
        super(FSMNUnit, self).__init__()

        self.shrink = LinearTransform(dimlinear, dimproj)
        self.fsmn = Fsmn(dimproj, dimproj, lorder, rorder, 1, 1)
        self.expand = AffineTransform(dimproj, dimlinear)

        self.debug = False
        self.dataout = None

    '''
    batch, time, channel, feature
    '''

    def forward(self, input):
        if torch.cuda.is_available():
            out = torch.zeros(input.shape).cuda()
        else:
            out = torch.zeros(input.shape)

        for n in range(input.shape[2]):
            out1 = self.shrink(input[:, :, n, :])
            out2 = self.fsmn(out1)
            out[:, :, n, :] = F.relu(self.expand(out2))

        if self.debug:
            self.dataout = out

        return out

    def print_model(self):
        self.shrink.print_model()
        self.fsmn.print_model()
        self.expand.print_model()

    def to_kaldi_nnet(self):
        re_str = self.shrink.to_kaldi_nnet()
        re_str += self.fsmn.to_kaldi_nnet()
        re_str += self.expand.to_kaldi_nnet()

        relu = RectifiedLinear(self.expand.linear.out_features,
                               self.expand.linear.out_features)
        re_str += relu.to_kaldi_nnet()

        return re_str


class FSMNSeleNetV2(nn.Module):
    """ FSMN model with channel selection.
    """

    def __init__(self,
                 input_dim=120,
                 linear_dim=128,
                 proj_dim=64,
                 lorder=20,
                 rorder=1,
                 num_syn=5,
                 fsmn_layers=5,
                 sele_layer=0):
        """
        Args:
            input_dim:              input dimension
            linear_dim:             fsmn input dimension
            proj_dim:               fsmn projection dimension
            lorder:                 fsmn left order
            rorder:                 fsmn right order
            num_syn:                output dimension
            fsmn_layers:            no. of fsmn units
            sele_layer:             channel selection layer index
        """
        super(FSMNSeleNetV2, self).__init__()

        self.sele_layer = sele_layer

        self.featmap = AffineTransform(input_dim, linear_dim)

        self.mem = []
        for i in range(fsmn_layers):
            unit = FSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(i), unit)

        self.decision = AffineTransform(linear_dim, num_syn)

    def forward(self, input):
        # multi-channel feature mapping
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.featmap.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.featmap.linear.out_features)

        for n in range(input.shape[2]):
            x[:, :, n, :] = F.relu(self.featmap(input[:, :, n, :]))

        for i, unit in enumerate(self.mem):
            y = unit(x)

            # perform channel selection
            if i == self.sele_layer:
                pool = nn.MaxPool2d((y.shape[2], 1), stride=(y.shape[2], 1))
                y = pool(y)

            x = y

        # remove channel dimension
        y = torch.squeeze(y, -2)
        z = self.decision(y)

        return z

    def print_model(self):
        self.featmap.print_model()

        for unit in self.mem:
            unit.print_model()

        self.decision.print_model()

    def print_header(self):
        '''
        get FSMN params
        '''
        input_dim = self.featmap.linear.in_features
        linear_dim = self.featmap.linear.out_features
        proj_dim = self.mem[0].shrink.linear.out_features
        lorder = self.mem[0].fsmn.conv_left.kernel_size[0]
        rorder = 0
        if self.mem[0].fsmn.conv_right is not None:
            rorder = self.mem[0].fsmn.conv_right.kernel_size[0]

        num_syn = self.decision.linear.out_features
        fsmn_layers = len(self.mem)

        # no. of output channels, 0.0 means the same as numins
        # numouts = 0.0
        numouts = 1.0

        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = numouts
        # dimins
        header[2] = input_dim
        # dimouts
        header[3] = num_syn
        # numlayers
        header[4] = 3

        #
        # write each layer's header
        #
        hidx = 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(
            ActivationType.ACTIVATION_RELU.value)
        hidx += 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = lorder
        header[HEADER_BLOCK_SIZE * hidx + 5] = rorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = fsmn_layers
        if numouts == 1.0:
            header[HEADER_BLOCK_SIZE * hidx + 7] = float(self.sele_layer)
        else:
            header[HEADER_BLOCK_SIZE * hidx + 7] = -1.0
        hidx += 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = numouts
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(
            ActivationType.ACTIVATION_SOFTMAX.value)

        for h in header:
            print(f32ToI32(h))

    def to_kaldi_nnet(self):
        re_str = '<Nnet>\n'

        re_str = self.featmap.to_kaldi_nnet()

        relu = RectifiedLinear(self.featmap.linear.out_features,
                               self.featmap.linear.out_features)
        re_str += relu.to_kaldi_nnet()

        for unit in self.mem:
            re_str += unit.to_kaldi_nnet()

        re_str += self.decision.to_kaldi_nnet()

        re_str += '<Softmax> %d %d\n' % (self.decision.linear.out_features,
                                         self.decision.linear.out_features)
        re_str += '<!EndOfComponent>\n'
        re_str += '</Nnet>\n'

        return re_str
