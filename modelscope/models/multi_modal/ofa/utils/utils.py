# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import torch


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
