# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import struct
from enum import Enum

HEADER_BLOCK_SIZE = 10


class LayerType(Enum):
    LAYER_DENSE = 1
    LAYER_GRU = 2
    LAYER_ATTENTION = 3
    LAYER_FSMN = 4
    LAYER_SEQUENTIAL_FSMN = 5
    LAYER_FSMN_SELE = 6
    LAYER_GRU_ATTENTION = 7
    LAYER_DFSMN = 8


class ActivationType(Enum):
    ACTIVATION_NONE = 0
    ACTIVATION_RELU = 1
    ACTIVATION_TANH = 2
    ACTIVATION_SIGMOID = 3
    ACTIVATION_SOFTMAX = 4
    ACTIVATION_LOGSOFTMAX = 5


def f32ToI32(f):
    """
    print layer
    """
    bs = struct.pack('f', f)

    ba = bytearray()
    ba.append(bs[0])
    ba.append(bs[1])
    ba.append(bs[2])
    ba.append(bs[3])

    return struct.unpack('i', ba)[0]


def printNeonMatrix(w):
    """
    print matrix with neon padding
    """
    numrows, numcols = w.shape
    numnecols = math.ceil(numcols / 4)

    for i in range(numrows):
        for j in range(numcols):
            print(f32ToI32(w[i, j]))

        for j in range(numnecols * 4 - numcols):
            print(0)


def printNeonVector(b):
    """
    print vector with neon padding
    """
    size = b.shape[0]
    nesize = math.ceil(size / 4)

    for i in range(size):
        print(f32ToI32(b[i]))

    for i in range(nesize * 4 - size):
        print(0)


def printDense(layer):
    """
    save dense layer
    """
    statedict = layer.state_dict()
    printNeonMatrix(statedict['weight'])
    printNeonVector(statedict['bias'])


def printGRU(layer):
    """
    save gru layer
    """
    statedict = layer.state_dict()
    weight = [statedict['weight_ih_l0'], statedict['weight_hh_l0']]
    bias = [statedict['bias_ih_l0'], statedict['bias_hh_l0']]
    numins, numouts = weight[0].shape
    numins = numins // 3

    # output input weights
    w_rx = weight[0][:numins, :]
    w_zx = weight[0][numins:numins * 2, :]
    w_x = weight[0][numins * 2:, :]
    printNeonMatrix(w_zx)
    printNeonMatrix(w_rx)
    printNeonMatrix(w_x)

    # output recurrent weights
    w_rh = weight[1][:numins, :]
    w_zh = weight[1][numins:numins * 2, :]
    w_h = weight[1][numins * 2:, :]
    printNeonMatrix(w_zh)
    printNeonMatrix(w_rh)
    printNeonMatrix(w_h)

    # output input bias
    b_rx = bias[0][:numins]
    b_zx = bias[0][numins:numins * 2]
    b_x = bias[0][numins * 2:]
    printNeonVector(b_zx)
    printNeonVector(b_rx)
    printNeonVector(b_x)

    # output recurrent bias
    b_rh = bias[1][:numins]
    b_zh = bias[1][numins:numins * 2]
    b_h = bias[1][numins * 2:]
    printNeonVector(b_zh)
    printNeonVector(b_rh)
    printNeonVector(b_h)
