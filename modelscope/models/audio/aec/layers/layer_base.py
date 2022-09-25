# Copyright (c) Alibaba, Inc. and its affiliates.

import abc
import re

import numpy as np
import torch.nn as nn


def expect_token_number(instr, token):
    first_token = re.match(r'^\s*' + token, instr)
    if first_token is None:
        return None
    instr = instr[first_token.end():]
    lr = re.match(r'^\s*(-?\d+\.?\d*e?-?\d*?)', instr)
    if lr is None:
        return None
    return instr[lr.end():], lr.groups()[0]


def expect_kaldi_matrix(instr):
    pos2 = instr.find('[', 0)
    pos3 = instr.find(']', pos2)
    mat = []
    for stt in instr[pos2 + 1:pos3].split('\n'):
        tmp_mat = np.fromstring(stt, dtype=np.float32, sep=' ')
        if tmp_mat.size > 0:
            mat.append(tmp_mat)
    return instr[pos3 + 1:], np.array(mat)


def to_kaldi_matrix(np_mat):
    """
    function that transform as str numpy mat to standard kaldi str matrix
    :param np_mat: numpy mat
    :return: str
    """
    np.set_printoptions(threshold=np.inf, linewidth=np.nan, suppress=True)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


class LayerBase(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super(LayerBase, self).__init__()

    @abc.abstractmethod
    def to_kaldi_nnet(self):
        pass
