# Copyright (c) Alibaba, Inc. and its affiliates.

import torch as th
import torch.nn as nn

from modelscope.models.audio.ans.layers.layer_base import (LayerBase,
                                                           to_kaldi_matrix)
from modelscope.utils.audio.audio_utils import (expect_kaldi_matrix,
                                                expect_token_number)


class AffineTransform(LayerBase):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.linear(input)

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

        return re_str

    def load_kaldi_nnet(self, instr):

        output = expect_token_number(
            instr,
            '<LearnRateCoef>',
        )
        if output is None:
            raise Exception('AffineTransform format error')

        instr, lr = output

        output = expect_token_number(instr, '<BiasLearnRateCoef>')
        if output is None:
            raise Exception('AffineTransform format error')

        instr, lr = output

        output = expect_token_number(instr, '<MaxNorm>')
        if output is None:
            raise Exception('AffineTransform format error')

        instr, lr = output

        output = expect_kaldi_matrix(instr)

        if output is None:
            raise Exception('AffineTransform format error')

        instr, mat = output

        self.linear.weight = th.nn.Parameter(
            th.from_numpy(mat).type(th.FloatTensor))

        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('AffineTransform format error')

        instr, mat = output
        self.linear.bias = th.nn.Parameter(
            th.from_numpy(mat).type(th.FloatTensor))
        return instr
