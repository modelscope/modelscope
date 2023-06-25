# ------------------------------------------------------------------------------
# Part of implementation is adopted from ProxylessNAS,
# made publicly available under the Apache License 2.0 at https://github.com/mit-han-lab/proxylessnas.
# ------------------------------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .layers import (IdentityLayer, LinearMixConvLayer, MBInvertedConvLayer,
                     MBInvertedMHSALayer, MBInvertedMixConvLayer,
                     MBInvertedRepConvLayer, SELayer, ZeroLayer)


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


def conv_func_by_name(name):
    name2ops = {
        'Identity':
        lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Zero':
        lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    name2ops.update({
        '3x3_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
        #######################################################################################
        '13_MixConv1':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 1
                                                      ),
        '13_MixConv2':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 2
                                                      ),
        '13_MixConv3':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 3
                                                      ),
        '13_MixConv4':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 4
                                                      ),
        '13_MixConv5':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 5
                                                      ),
        '13_MixConv6':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 6
                                                      ),
        #######################################################################################
        '35_MixConv1':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 1
                                                      ),
        '35_MixConv2':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 2
                                                      ),
        '35_MixConv3':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 3
                                                      ),
        '35_MixConv4':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 4
                                                      ),
        '35_MixConv5':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 5
                                                      ),
        '35_MixConv6':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 6
                                                      ),
        #######################################################################################
        '135_MixConv1':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 1),
        '135_MixConv2':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 2),
        '135_MixConv3':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 3),
        '135_MixConv4':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 4),
        '135_MixConv5':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 5),
        '135_MixConv6':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 6),
        #######################################################################################
        '13_LinMixConv':
        lambda in_C, out_C, S: LinearMixConvLayer(in_C, out_C, [1, 3], S),
        '35_LinMixConv':
        lambda in_C, out_C, S: LinearMixConvLayer(in_C, out_C, [3, 5], S),
        '135_LinMixConv':
        lambda in_C, out_C, S: LinearMixConvLayer(in_C, out_C, [1, 3, 5], S),
        #######################################################################################
        'SE_2':
        lambda in_C, out_C, S: SELayer(in_C, 2),
        'SE_4':
        lambda in_C, out_C, S: SELayer(in_C, 4),
        'SE_8':
        lambda in_C, out_C, S: SELayer(in_C, 8),
        #######################################################################################
        'MHSA1':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 1, width, height),
        'MHSA2':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 2, width, height),
        'MHSA3':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 3, width, height),
        'MHSA4':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 4, width, height),
        'MHSA5':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 5, width, height),
        'MHSA6':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 6, width, height),
        #######################################################################################
        '13_RepConv1':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 1
                                                      ),
        '13_RepConv2':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 2
                                                      ),
        '13_RepConv3':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 3
                                                      ),
        '13_RepConv4':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 4
                                                      ),
        '13_RepConv5':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 5
                                                      ),
        '13_RepConv6':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 6
                                                      ),
        #######################################################################################
        '35_RepConv1':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 1
                                                      ),
        '35_RepConv2':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 2
                                                      ),
        '35_RepConv3':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 3
                                                      ),
        '35_RepConv4':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 4
                                                      ),
        '35_RepConv5':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 5
                                                      ),
        '35_RepConv6':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 6
                                                      ),
        #######################################################################################
        '135_RepConv1':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 1),
        '135_RepConv2':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 2),
        '135_RepConv3':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 3),
        '135_RepConv4':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 4),
        '135_RepConv5':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 5),
        '135_RepConv6':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 6),
    })
    return name2ops[name]


def build_candidate_ops(candidate_ops,
                        in_channels,
                        out_channels,
                        stride,
                        ops_order,
                        spatio_size=None):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        'Identity':
        lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Zero':
        lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add MBConv layers
    name2ops.update({
        '3x3_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6':
        lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
        #######################################################################################
        '13_MixConv1':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 1
                                                      ),
        '13_MixConv2':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 2
                                                      ),
        '13_MixConv3':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 3
                                                      ),
        '13_MixConv4':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 4
                                                      ),
        '13_MixConv5':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 5
                                                      ),
        '13_MixConv6':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3], S, 6
                                                      ),
        #######################################################################################
        '35_MixConv1':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 1
                                                      ),
        '35_MixConv2':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 2
                                                      ),
        '35_MixConv3':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 3
                                                      ),
        '35_MixConv4':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 4
                                                      ),
        '35_MixConv5':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 5
                                                      ),
        '35_MixConv6':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [3, 5], S, 6
                                                      ),
        #######################################################################################
        '135_MixConv1':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 1),
        '135_MixConv2':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 2),
        '135_MixConv3':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 3),
        '135_MixConv4':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 4),
        '135_MixConv5':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 5),
        '135_MixConv6':
        lambda in_C, out_C, S: MBInvertedMixConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 6),
        #######################################################################################
        '13_LinMixConv':
        lambda in_C, out_C, S: LinearMixConvLayer(in_C, out_C, [1, 3], S),
        '35_LinMixConv':
        lambda in_C, out_C, S: LinearMixConvLayer(in_C, out_C, [3, 5], S),
        '135_LinMixConv':
        lambda in_C, out_C, S: LinearMixConvLayer(in_C, out_C, [1, 3, 5], S),
        #######################################################################################
        'SE_2':
        lambda in_C, out_C, S: SELayer(in_C, 2),
        'SE_4':
        lambda in_C, out_C, S: SELayer(in_C, 4),
        'SE_8':
        lambda in_C, out_C, S: SELayer(in_C, 8),
        #######################################################################################
        'MHSA1':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 1, width, height),
        'MHSA2':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 2, width, height),
        'MHSA3':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 3, width, height),
        'MHSA4':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 4, width, height),
        'MHSA5':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 5, width, height),
        'MHSA6':
        lambda in_C, out_C, width, height: MBInvertedMHSALayer(
            in_C, out_C, 6, width, height),
        #######################################################################################
        '13_RepConv1':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 1
                                                      ),
        '13_RepConv2':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 2
                                                      ),
        '13_RepConv3':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 3
                                                      ),
        '13_RepConv4':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 4
                                                      ),
        '13_RepConv5':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 5
                                                      ),
        '13_RepConv6':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3], S, 6
                                                      ),
        #######################################################################################
        '35_RepConv1':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 1
                                                      ),
        '35_RepConv2':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 2
                                                      ),
        '35_RepConv3':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 3
                                                      ),
        '35_RepConv4':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 4
                                                      ),
        '35_RepConv5':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 5
                                                      ),
        '35_RepConv6':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [3, 5], S, 6
                                                      ),
        #######################################################################################
        '135_RepConv1':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 1),
        '135_RepConv2':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 2),
        '135_RepConv3':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 3),
        '135_RepConv4':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 4),
        '135_RepConv5':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 5),
        '135_RepConv6':
        lambda in_C, out_C, S: MBInvertedRepConvLayer(in_C, out_C, [1, 3, 5],
                                                      S, 6),
    })

    out = []
    for name in candidate_ops:

        if 'MHSA' in name:
            out.append(name2ops[name](in_channels, out_channels,
                                      spatio_size[0], spatio_size[1]))
        else:
            out.append(name2ops[name](in_channels, out_channels, stride))
    return out


class MixedEdge(nn.Module):
    MODE = None  # full, two, None, full_v2

    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(
            self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(torch.Tensor(
            self.n_choices))  # binary gates

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def forward(self, x):
        if MixedEdge.MODE == 'full' or MixedEdge.MODE == 'two':
            output = 0
            for _i in self.active_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi
            for _i in self.inactive_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi.detach()
        elif MixedEdge.MODE == 'full_v2':

            def run_function(candidate_ops, active_id):

                def forward(_x):
                    return candidate_ops[active_id](_x)

                return forward

            def backward_function(candidate_ops, active_id, binary_gates):

                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                out_k = candidate_ops[k](_x.data)
                            else:
                                out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[k] = grad_k
                    return binary_grads

                return backward

            output = ArchGradientFunction.apply(
                x, self.AP_path_wb,
                run_function(self.candidate_ops, self.active_index[0]),
                backward_function(self.candidate_ops, self.active_index[0],
                                  self.AP_path_wb))
        else:
            output = self.active_op(x)
        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str,
                                  probs)

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        if MixedEdge.MODE == 'two':
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(
                torch.stack([self.AP_path_alpha[idx] for idx in sample_op]),
                dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i]
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0
        else:
            sample = torch.multinomial(probs.data, 1)[0].item()
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        binary_grads = self.AP_path_wb.grad.data
        if self.active_op.is_zero_layer():
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if MixedEdge.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(
                torch.stack([self.AP_path_alpha[idx] for idx in involved_idx]),
                dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx,
                                         self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx,
                                           self.AP_path_alpha.data[idx].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[
                        i] += binary_grads[j] * probs[j] * (
                            delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [
            idx for idx, _ in (self.active_index + self.inactive_index)
        ]
        old_alphas = [
            alpha for _, alpha in (self.active_index + self.inactive_index)
        ]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas])
            / sum([math.exp(alpha) for alpha in old_alphas]))

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(
            output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data,
                                         grad_output.data)

        return grad_x[0], binary_grads, None, None
