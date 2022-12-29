'''
FSMN implementation.

Copyright: 2022-03-09 yueyue.nyy
'''

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def toKaldiMatrix(np_mat):
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


def printTensor(torch_tensor):
    re_str = ''
    x = torch_tensor.detach().squeeze().numpy()
    re_str += toKaldiMatrix(x)
    # re_str += '<!EndOfComponent>\n'
    print(re_str)


class LinearTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        output = self.quant(input)
        output = self.linear(output)
        output = self.dequant(output)

        return output

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<LinearTransform> %d %d\n' % (self.output_dim,
                                                 self.input_dim)
        re_str += '<LearnRateCoef> 1\n'

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += toKaldiMatrix(x)
        # re_str += '<!EndOfComponent>\n'

        return re_str

    def to_pytorch_net(self, fread):
        linear_line = fread.readline()
        linear_split = linear_line.strip().split()
        assert len(linear_split) == 3
        assert linear_split[0] == '<LinearTransform>'
        self.output_dim = int(linear_split[1])
        self.input_dim = int(linear_split[2])

        learn_rate_line = fread.readline()
        assert learn_rate_line.find('LearnRateCoef') != -1

        self.linear.reset_parameters()

        # linear_weights = self.state_dict()['linear.weight']
        # print(linear_weights.shape)
        new_weights = torch.zeros((self.output_dim, self.input_dim),
                                  dtype=torch.float32)
        for i in range(self.output_dim):
            line = fread.readline()
            splits = line.strip().strip('[]').strip().split()
            assert len(splits) == self.input_dim
            cols = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)
            new_weights[i, :] = cols

        self.linear.weight.data = new_weights


class AffineTransform(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        output = self.quant(input)
        output = self.linear(output)
        output = self.dequant(output)

        return output

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<AffineTransform> %d %d\n' % (self.output_dim,
                                                 self.input_dim)
        re_str += '<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0\n'

        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += toKaldiMatrix(x)

        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += toKaldiMatrix(x)
        # re_str += '<!EndOfComponent>\n'

        return re_str

    def to_pytorch_net(self, fread):
        affine_line = fread.readline()
        affine_split = affine_line.strip().split()
        assert len(affine_split) == 3
        assert affine_split[0] == '<AffineTransform>'
        self.output_dim = int(affine_split[1])
        self.input_dim = int(affine_split[2])
        print('AffineTransform output/input dim: %d %d' %
              (self.output_dim, self.input_dim))

        learn_rate_line = fread.readline()
        assert learn_rate_line.find('LearnRateCoef') != -1

        # linear_weights = self.state_dict()['linear.weight']
        # print(linear_weights.shape)
        self.linear.reset_parameters()

        new_weights = torch.zeros((self.output_dim, self.input_dim),
                                  dtype=torch.float32)
        for i in range(self.output_dim):
            line = fread.readline()
            splits = line.strip().strip('[]').strip().split()
            assert len(splits) == self.input_dim
            cols = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)
            new_weights[i, :] = cols

        self.linear.weight.data = new_weights

        # linear_bias = self.state_dict()['linear.bias']
        # print(linear_bias.shape)
        bias_line = fread.readline()
        splits = bias_line.strip().strip('[]').strip().split()
        assert len(splits) == self.output_dim
        new_bias = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)

        self.linear.bias.data = new_bias


class FSMNBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lorder=None,
        rorder=None,
        lstride=1,
        rstride=1,
    ):
        super(FSMNBlock, self).__init__()

        self.dim = input_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride

        self.conv_left = nn.Conv2d(
            self.dim,
            self.dim, [lorder, 1],
            dilation=[lstride, 1],
            groups=self.dim,
            bias=False)

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim,
                self.dim, [rorder, 1],
                dilation=[rstride, 1],
                groups=self.dim,
                bias=False)
        else:
            self.conv_right = None

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)

        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])
        y_left = self.quant(y_left)
        y_left = self.conv_left(y_left)
        y_left = self.dequant(y_left)
        out = x_per + y_left

        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            y_right = self.quant(y_right)
            y_right = self.conv_right(y_right)
            y_right = self.dequant(y_right)
            out += y_right

        out_per = out.permute(0, 3, 2, 1)
        output = out_per.squeeze(1)

        return output

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<Fsmn> %d %d\n' % (self.dim, self.dim)
        re_str += '<LearnRateCoef> %d <LOrder> %d <ROrder> %d <LStride> %d <RStride> %d <MaxNorm> 0\n' % (
            1, self.lorder, self.rorder, self.lstride, self.rstride)

        # print(self.conv_left.weight,self.conv_right.weight)
        lfiters = self.state_dict()['conv_left.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += toKaldiMatrix(x)

        if self.conv_right is not None:
            rfiters = self.state_dict()['conv_right.weight']
            x = (rfiters.squeeze().numpy().T)
            re_str += toKaldiMatrix(x)
            # re_str += '<!EndOfComponent>\n'

        return re_str

    def to_pytorch_net(self, fread):
        fsmn_line = fread.readline()
        fsmn_split = fsmn_line.strip().split()
        assert len(fsmn_split) == 3
        assert fsmn_split[0] == '<Fsmn>'
        self.dim = int(fsmn_split[1])

        params_line = fread.readline()
        params_split = params_line.strip().strip('[]').strip().split()
        assert len(params_split) == 12
        assert params_split[0] == '<LearnRateCoef>'
        assert params_split[2] == '<LOrder>'
        self.lorder = int(params_split[3])
        assert params_split[4] == '<ROrder>'
        self.rorder = int(params_split[5])
        assert params_split[6] == '<LStride>'
        self.lstride = int(params_split[7])
        assert params_split[8] == '<RStride>'
        self.rstride = int(params_split[9])
        assert params_split[10] == '<MaxNorm>'

        # lfilters = self.state_dict()['conv_left.weight']
        # print(lfilters.shape)
        print('read conv_left weight')
        new_lfilters = torch.zeros((self.lorder, 1, self.dim, 1),
                                   dtype=torch.float32)
        for i in range(self.lorder):
            print('read conv_left weight -- %d' % i)
            line = fread.readline()
            splits = line.strip().strip('[]').strip().split()
            assert len(splits) == self.dim
            cols = torch.tensor([float(item) for item in splits],
                                dtype=torch.float32)
            new_lfilters[self.lorder - 1 - i, 0, :, 0] = cols

        new_lfilters = torch.transpose(new_lfilters, 0, 2)
        # print(new_lfilters.shape)

        self.conv_left.reset_parameters()
        self.conv_left.weight.data = new_lfilters
        # print(self.conv_left.weight.shape)

        if self.rorder > 0:
            # rfilters = self.state_dict()['conv_right.weight']
            # print(rfilters.shape)
            print('read conv_right weight')
            new_rfilters = torch.zeros((self.rorder, 1, self.dim, 1),
                                       dtype=torch.float32)
            line = fread.readline()
            for i in range(self.rorder):
                print('read conv_right weight -- %d' % i)
                line = fread.readline()
                splits = line.strip().strip('[]').strip().split()
                assert len(splits) == self.dim
                cols = torch.tensor([float(item) for item in splits],
                                    dtype=torch.float32)
                new_rfilters[i, 0, :, 0] = cols

            new_rfilters = torch.transpose(new_rfilters, 0, 2)
            # print(new_rfilters.shape)
            self.conv_right.reset_parameters()
            self.conv_right.weight.data = new_rfilters
            # print(self.conv_right.weight.shape)


class RectifiedLinear(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.relu(input)
        # out = self.dropout(out)
        return out

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<RectifiedLinear> %d %d\n' % (self.dim, self.dim)
        # re_str += '<!EndOfComponent>\n'
        return re_str

        # re_str = ''
        # re_str += '<ParametricRelu> %d %d\n' % (self.dim, self.dim)
        # re_str += '<AlphaLearnRateCoef> 0 <BetaLearnRateCoef> 0\n'
        # re_str += toKaldiMatrix(np.ones((self.dim), dtype = 'int32'))
        # re_str += toKaldiMatrix(np.zeros((self.dim), dtype = 'int32'))
        # re_str += '<!EndOfComponent>\n'
        # return re_str

    def to_pytorch_net(self, fread):
        line = fread.readline()
        splits = line.strip().split()
        assert len(splits) == 3
        assert splits[0] == '<RectifiedLinear>'
        assert int(splits[1]) == int(splits[2])
        assert int(splits[1]) == self.dim
        self.dim = int(splits[1])


def _build_repeats(
    fsmn_layers: int,
    linear_dim: int,
    proj_dim: int,
    lorder: int,
    rorder: int,
    lstride=1,
    rstride=1,
):
    repeats = [
        nn.Sequential(
            LinearTransform(linear_dim, proj_dim),
            FSMNBlock(proj_dim, proj_dim, lorder, rorder, 1, 1),
            AffineTransform(proj_dim, linear_dim),
            RectifiedLinear(linear_dim, linear_dim))
        for i in range(fsmn_layers)
    ]

    return nn.Sequential(*repeats)


class FSMN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        input_affine_dim: int,
        fsmn_layers: int,
        linear_dim: int,
        proj_dim: int,
        lorder: int,
        rorder: int,
        lstride: int,
        rstride: int,
        output_affine_dim: int,
        output_dim: int,
    ):
        """
            Args:
                input_dim:              input dimension
                input_affine_dim:       input affine layer dimension
                fsmn_layers:            no. of fsmn units
                linear_dim:             fsmn input dimension
                proj_dim:               fsmn projection dimension
                lorder:                 fsmn left order
                rorder:                 fsmn right order
                lstride:                fsmn left stride
                rstride:                fsmn right stride
                output_affine_dim:      output affine layer dimension
                output_dim:             output dimension
        """
        super(FSMN, self).__init__()

        self.input_dim = input_dim
        self.input_affine_dim = input_affine_dim
        self.fsmn_layers = fsmn_layers
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        self.output_affine_dim = output_affine_dim
        self.output_dim = output_dim

        self.in_linear1 = AffineTransform(input_dim, input_affine_dim)
        self.in_linear2 = AffineTransform(input_affine_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)

        self.fsmn = _build_repeats(fsmn_layers, linear_dim, proj_dim, lorder,
                                   rorder, lstride, rstride)

        self.out_linear1 = AffineTransform(linear_dim, output_affine_dim)
        self.out_linear2 = AffineTransform(output_affine_dim, output_dim)
        # self.softmax = nn.Softmax(dim = -1)

    def fuse_modules(self):
        pass

    def forward(
        self,
        input: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
            in_cache(torhc.Tensor): (B, D, C), C is the accumulated cache size
        """

        # print("FSMN forward!!!!")
        # print(input.shape)
        # print(input)
        # print(self.in_linear1.input_dim)
        # print(self.in_linear1.output_dim)

        x1 = self.in_linear1(input)
        x2 = self.in_linear2(x1)
        x3 = self.relu(x2)
        x4 = self.fsmn(x3)
        x5 = self.out_linear1(x4)
        x6 = self.out_linear2(x5)
        # x7 = self.softmax(x6)

        # return x7, None
        return x6, in_cache

    def to_kaldi_net(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.in_linear1.to_kaldi_net()
        re_str += self.in_linear2.to_kaldi_net()
        re_str += self.relu.to_kaldi_net()

        for fsmn in self.fsmn:
            re_str += fsmn[0].to_kaldi_net()
            re_str += fsmn[1].to_kaldi_net()
            re_str += fsmn[2].to_kaldi_net()
            re_str += fsmn[3].to_kaldi_net()

        re_str += self.out_linear1.to_kaldi_net()
        re_str += self.out_linear2.to_kaldi_net()
        re_str += '<Softmax> %d %d\n' % (self.output_dim, self.output_dim)
        # re_str += '<!EndOfComponent>\n'
        re_str += '</Nnet>\n'

        return re_str

    def to_pytorch_net(self, kaldi_file):
        with open(kaldi_file, 'r', encoding='utf8') as fread:
            fread = open(kaldi_file, 'r')
            nnet_start_line = fread.readline()
            assert nnet_start_line.strip() == '<Nnet>'

            self.in_linear1.to_pytorch_net(fread)
            self.in_linear2.to_pytorch_net(fread)
            self.relu.to_pytorch_net(fread)

            for fsmn in self.fsmn:
                fsmn[0].to_pytorch_net(fread)
                fsmn[1].to_pytorch_net(fread)
                fsmn[2].to_pytorch_net(fread)
                fsmn[3].to_pytorch_net(fread)

            self.out_linear1.to_pytorch_net(fread)
            self.out_linear2.to_pytorch_net(fread)

            softmax_line = fread.readline()
            softmax_split = softmax_line.strip().split()
            assert softmax_split[0].strip() == '<Softmax>'
            assert int(softmax_split[1]) == self.output_dim
            assert int(softmax_split[2]) == self.output_dim
            # '<!EndOfComponent>\n'

            nnet_end_line = fread.readline()
            assert nnet_end_line.strip() == '</Nnet>'
        fread.close()


if __name__ == '__main__':
    fsmn = FSMN(400, 140, 4, 250, 128, 10, 2, 1, 1, 140, 2599)
    print(fsmn)

    num_params = sum(p.numel() for p in fsmn.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(128, 200, 400)  # batch-size * time * dim
    y, _ = fsmn(x)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))

    print(fsmn.to_kaldi_net())
