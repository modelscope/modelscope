import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .layers import (LinearMixConvLayer, MBInvertedConvLayer,
                     MBInvertedMixConvLayer, MBInvertedRepConvLayer, SELayer,
                     ZeroLayer)


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
        lambda in_C, out_C, S, height: IdentityLayer(
            in_C, out_C, ops_order=ops_order),
        'Zero':
        lambda in_C, out_C, S, height: ZeroLayer(stride=S),
    }
    name2ops.update({
        '3x3_MBConv1':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 3), 3), S, 1),
        '3x3_MBConv2':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 3), 3), S, 2),
        '3x3_MBConv3':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 3), 3), S, 3),
        '3x3_MBConv4':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 3), 3), S, 4),
        '3x3_MBConv5':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 3), 3), S, 5),
        '3x3_MBConv6':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 3), 3), S, 6),
        #######################################################################################
        '5x5_MBConv1':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 5), 5), S, 1),
        '5x5_MBConv2':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 5), 5), S, 2),
        '5x5_MBConv3':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 5), 5), S, 3),
        '5x5_MBConv4':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 5), 5), S, 4),
        '5x5_MBConv5':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 5), 5), S, 5),
        '5x5_MBConv6':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 5), 5), S, 6),
        #######################################################################################
        '7x7_MBConv1':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 7), 7), S, 1),
        '7x7_MBConv2':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 7), 7), S, 2),
        '7x7_MBConv3':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 7), 7), S, 3),
        '7x7_MBConv4':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 7), 7), S, 4),
        '7x7_MBConv5':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 7), 7), S, 5),
        '7x7_MBConv6':
        lambda in_C, out_C, S, height: MBInvertedConvLayer(
            in_C, out_C, (min(height, 7), 7), S, 6),
        #######################################################################################
        '13_MixConv1':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 1),
        '13_MixConv2':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 2),
        '13_MixConv3':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 3),
        '13_MixConv4':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 4),
        '13_MixConv5':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 5),
        '13_MixConv6':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 6),
        #######################################################################################
        '35_MixConv1':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 1),
        '35_MixConv2':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 2),
        '35_MixConv3':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 3),
        '35_MixConv4':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 4),
        '35_MixConv5':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 5),
        '35_MixConv6':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 6),
        #######################################################################################
        '135_MixConv1':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 1),
        '135_MixConv2':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 2),
        '135_MixConv3':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 3),
        '135_MixConv4':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 4),
        '135_MixConv5':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 5),
        '135_MixConv6':
        lambda in_C, out_C, S, height: MBInvertedMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 6),
        #######################################################################################
        '13_LinMixConv':
        lambda in_C, out_C, S, height: LinearMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S),
        '35_LinMixConv':
        lambda in_C, out_C, S, height: LinearMixConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S),
        '135_LinMixConv':
        lambda in_C, out_C, S, height: LinearMixConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S),
        #######################################################################################
        'SE_2':
        lambda in_C, out_C, S, height: SELayer(in_C, 2),
        'SE_4':
        lambda in_C, out_C, S, height: SELayer(in_C, 4),
        'SE_8':
        lambda in_C, out_C, S, height: SELayer(in_C, 8),
        #######################################################################################
        '13_RepConv1':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 1),
        '13_RepConv2':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 2),
        '13_RepConv3':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 3),
        '13_RepConv4':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 4),
        '13_RepConv5':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 5),
        '13_RepConv6':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3)], S, 6),
        #######################################################################################
        '35_RepConv1':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 1),
        '35_RepConv2':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 2),
        '35_RepConv3':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 3),
        '35_RepConv4':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 4),
        '35_RepConv5':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 5),
        '35_RepConv6':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [(min(height, 3), 3), (min(height, 5), 5)], S, 6),
        #######################################################################################
        '135_RepConv1':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 1),
        '135_RepConv2':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 2),
        '135_RepConv3':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 3),
        '135_RepConv4':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 4),
        '135_RepConv5':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 5),
        '135_RepConv6':
        lambda in_C, out_C, S, height: MBInvertedRepConvLayer(
            in_C, out_C, [1, (min(height, 3), 3), (min(height, 5), 5)], S, 6),
    })
    return name2ops[name]
