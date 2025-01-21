# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn

from .layer_base import LayerBase


class RectifiedLinear(LayerBase):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(input)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<RectifiedLinear> %d %d\n' % (self.dim, self.dim)
        return re_str

    def load_kaldi_nnet(self, instr):
        return instr


class LogSoftmax(LayerBase):

    def __init__(self, input_dim, output_dim):
        super(LogSoftmax, self).__init__()
        self.dim = input_dim
        self.ls = nn.LogSoftmax()

    def forward(self, input):
        return self.ls(input)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<Softmax> %d %d\n' % (self.dim, self.dim)
        return re_str

    def load_kaldi_nnet(self, instr):
        return instr


class Sigmoid(LayerBase):

    def __init__(self, input_dim, output_dim):
        super(Sigmoid, self).__init__()
        self.dim = input_dim
        self.sig = nn.Sigmoid()

    def forward(self, input):
        return self.sig(input)

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<Sigmoid> %d %d\n' % (self.dim, self.dim)
        return re_str

    def load_kaldi_nnet(self, instr):
        return instr
