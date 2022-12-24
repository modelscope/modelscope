# Copyright (c) Alibaba, Inc. and its affiliates.

from distutils.version import LooseVersion

import torch

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion('1.7')


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = (
        torch.arange(0, max_len).unsqueeze(0).expand(batch_size,
                                                     -1).to(lengths.device))
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
