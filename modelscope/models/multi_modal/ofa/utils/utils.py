# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import torch
import torch.nn as nn


def expand_mask(mask: torch.Tensor,
                dtype: torch.dtype,
                tgt_len: Optional[int] = None):
    r"""
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                  src_len).to(dtype)
    return expanded_mask.masked_fill(expanded_mask.bool(),
                                     torch.finfo(dtype).min)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    r"""
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        x (`nn.Modules`): input nn layers.
        drop_prob (`float`): drop path ratio.
        training (`bool`): whether is training or inference.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (1, x.shape[1], 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    r"""
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    See more details about drop path from https://arxiv.org/pdf/1605.07648v4.pdf.

    Args:
        drop_prob: drop path ratio.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
