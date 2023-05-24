# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsmn import AffineTransform, Fsmn, LinearTransform, RectifiedLinear
from .model_def import HEADER_BLOCK_SIZE, ActivationType, LayerType, f32ToI32


class DFSMNUnit(nn.Module):
    """ one multi-channel deep fsmn unit
    Args:
        dimin:                  input dimension
        dimexpand:              feature expansion dimension
        dimout:                 output dimension
        lorder:                 left ofder
        rorder:                 right order
    """

    def __init__(self,
                 dimin=64,
                 dimexpand=128,
                 dimout=64,
                 lorder=10,
                 rorder=1):
        super(DFSMNUnit, self).__init__()

        self.expand = AffineTransform(dimin, dimexpand)
        self.shrink = LinearTransform(dimexpand, dimout)
        self.fsmn = Fsmn(dimout, dimout, lorder, rorder, 1, 1)

        self.debug = False
        self.dataout = None

    def forward(self, input):
        """
        Args:
            input: [batch, time, feature]
        """
        out1 = F.relu(self.expand(input))
        out2 = self.shrink(out1)
        out3 = self.fsmn(out2)

        # add skip connection for matched data
        if input.shape[-1] == out3.shape[-1]:
            out3 = input + out3
        if self.debug:
            self.dataout = out3
        return out3

    def print_model(self):
        self.expand.printModel()
        self.shrink.printModel()
        self.fsmn.printModel()

    def to_kaldi_nnet(self):
        re_str = self.expand.toKaldiNNet()
        relu = RectifiedLinear(self.expand.linear.out_features,
                               self.expand.linear.out_features)
        re_str += relu.toKaldiNNet()
        re_str = self.shrink.toKaldiNNet()
        re_str += self.fsmn.toKaldiNNet()
        return re_str


class FSMNSeleNetV3(nn.Module):
    """ Deep FSMN model with channel selection performs multi-channel kws.
    Zhang, Shiliang, et al. "Deep-FSMN for large vocabulary continuous speech
    recognition." 2018 IEEE International Conference on Acoustics, Speech and
    Signal Processing (ICASSP). IEEE, 2018.

    Args:
        input_dim:              input dimension
        linear_dim:             fsmn input dimension
        proj_dim:               fsmn projection dimension
        lorder:                 fsmn left order
        rorder:                 fsmn right order
        num_syn:                output dimension
        fsmn_layers:            no. of fsmn units
    """

    def __init__(self,
                 input_dim=120,
                 linear_dim=128,
                 proj_dim=64,
                 lorder=10,
                 rorder=1,
                 num_syn=5,
                 fsmn_layers=5):
        super(FSMNSeleNetV3, self).__init__()

        self.mem = []
        # the first unit, mapping input dim to proj dim
        unit = DFSMNUnit(input_dim, linear_dim, proj_dim, lorder, rorder)
        self.mem.append(unit)
        self.add_module('mem_{:d}'.format(0), unit)

        # deep fsmn layers with skip connection
        for i in range(1, fsmn_layers):
            unit = DFSMNUnit(proj_dim, linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(i), unit)

        self.expand2 = AffineTransform(proj_dim, linear_dim)
        self.decision = AffineTransform(linear_dim, num_syn)

    def forward(self, input):
        # multi-channel temp space, [batch, time, channel, feature]
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.expand2.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2],
                            self.expand2.linear.out_features)

        for n in range(input.shape[2]):
            chin = input[:, :, n, :]

            for unit in self.mem:
                chout = unit(chin)
                chin = chout

            x[:, :, n, :] = F.relu(self.expand2(chout))

        # perform max pooling
        pool = nn.MaxPool2d((x.shape[2], 1), stride=(x.shape[2], 1))
        y = pool(x)

        # remove channel dimension
        y = torch.squeeze(y, -2)
        z = self.decision(y)

        return z

    def print_model(self):
        for unit in self.mem:
            unit.print_model()

        self.expand2.printModel()
        self.decision.printModel()

    def print_header(self):
        """ get DFSMN params
        """
        input_dim = self.mem[0].expand.linear.in_features
        linear_dim = self.mem[0].expand.linear.out_features
        proj_dim = self.mem[0].shrink.linear.out_features
        lorder = self.mem[0].fsmn.conv_left.kernel_size[0]
        rorder = 0
        if self.mem[0].fsmn.conv_right is not None:
            rorder = self.mem[0].fsmn.conv_right.kernel_size[0]

        num_syn = self.decision.linear.out_features
        fsmn_layers = len(self.mem)

        # no. of output channels, 0.0 means the same as numins
        numouts = 1.0

        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 5
        # numins
        header[0] = 0.0
        # numouts
        header[1] = numouts
        # dimins
        header[2] = input_dim
        # dimouts
        header[3] = num_syn
        # numlayers
        header[4] = 4

        #
        # write each layer's header
        #
        hidx = 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_DFSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 5] = lorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = rorder
        header[HEADER_BLOCK_SIZE * hidx + 7] = fsmn_layers
        hidx += 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(
            ActivationType.ACTIVATION_RELU.value)
        hidx += 1

        header[HEADER_BLOCK_SIZE * hidx + 0] = float(
            LayerType.LAYER_MAX_POOLING.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
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
        for unit in self.mem:
            re_str += unit.to_kaldi_nnet()
        re_str = self.expand2.toKaldiNNet()
        relu = RectifiedLinear(self.expand2.linear.out_features,
                               self.expand2.linear.out_features)
        re_str += relu.toKaldiNNet()
        re_str += self.decision.toKaldiNNet()
        re_str += '<Softmax> %d %d\n' % (self.decision.linear.out_features,
                                         self.decision.linear.out_features)
        re_str += '<!EndOfComponent>\n'
        re_str += '</Nnet>\n'

        return re_str
