# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license which can be found at
# https://github.com/facebookresearch/fairseq/blob/main/LICENSE

import collections
from collections import abc
from itertools import accumulate

import torch
import torch.nn.functional as F

try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

MANIFOLD_PATH_SEP = '|'


def apply_to_sample(f, sample):
    r"""
    Apply some function to the sample. The f function will effect on the `Tensor` object, otherwise do nothing.
    """
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_device(batch, device):
    r"""
    Puts each data field to the device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return tuple(move_to_device(item, device) for item in batch)
    elif isinstance(batch, abc.Mapping):
        return {
            key: move_to_device(value, device)
            for key, value in batch.items()
        }
    else:
        return batch


def strip_pad(tensor, pad):
    r"""
    Get the non pad value from input tensor
    """
    return tensor[tensor.ne(pad)]


def get_token_to_word_mapping(tokens, exclude_list):
    r"""
    Get the token to word mapping. The token indicates the original token index, while word indicates the token index
    excluding the `exclude_list`.

    >>> import torch
    >>> all_tokens = torch.arange(4)
    >>> exclude_tokens = [1]
    >>> get_token_to_word_mapping(all_tokens, exclude_tokens) # {0: 1, 1: 1, 2: 2, 3: 3}
    """
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    r"""
    @deprecated
    There is no usage in this project, should be removed.
    """
    tgt_valid = (((tgt_sent != pad) &  # noqa
                  (tgt_sent != eos)).nonzero(as_tuple=False).squeeze(dim=-1))
    src_invalid = (((src_sent == pad) |  # noqa
                    (src_sent == eos)).nonzero(as_tuple=False).squeeze(dim=-1))
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float('-inf')
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append((
                src_token_to_word[src_idx.item()] - 1,
                tgt_token_to_word[tgt_idx.item()] - 1,
            ))
    return alignment


def softmax(x, dim: int, onnx_trace: bool = False):
    r"""
    softmax function. Using `torch.nn.functional.softmax`
    """
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    r"""
    log softmax function. Using `torch.nn.functional.log_softmax`
    """
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def extract_soft_alignment(attn, src_sent, tgt_sent, pad, eos):
    r"""
    @deprecated
    There is no usage in this project, should be removed.
    """
    tgt_valid = (tgt_sent != pad).nonzero(as_tuple=False)
    src_valid = (src_sent != pad).nonzero(as_tuple=False).squeeze(dim=-1)
    alignment = []
    if len(tgt_valid) != 0 and len(src_valid) != 0:
        attn_valid = attn[tgt_valid, src_valid]
        alignment = [['{:.6f}'.format(p) for p in src_probs.tolist()]
                     for src_probs in attn_valid]
    return alignment
