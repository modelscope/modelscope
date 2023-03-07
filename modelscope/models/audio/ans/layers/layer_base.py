# Copyright (c) Alibaba, Inc. and its affiliates.

import abc

import numpy as np
import six
import torch.nn as nn


def to_kaldi_matrix(np_mat):
    """ function that transform as str numpy mat to standard kaldi str matrix

    Args:
        np_mat: numpy mat
    """
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


@six.add_metaclass(abc.ABCMeta)
class LayerBase(nn.Module):

    def __init__(self):
        super(LayerBase, self).__init__()

    @abc.abstractmethod
    def to_kaldi_nnet(self):
        pass
