# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
from collections.abc import Mapping


def torch_nested_numpify(tensors):
    """ Numpify nested torch tensors.

    NOTE: If the type of input tensors is dict-like(Mapping, dict, OrderedDict, etc.), the return type will be dict.

    Args:
        tensors: Nested torch tensors.

    Returns:
        The numpify tensors.
    """

    import torch
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(torch_nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        # return dict
        return {k: torch_nested_numpify(t) for k, t in tensors.items()}
    if isinstance(tensors, torch.Tensor):
        t = tensors.cpu()
        return t.numpy()
    return tensors


def torch_nested_detach(tensors):
    """ Detach nested torch tensors.

    NOTE: If the type of input tensors is dict-like(Mapping, dict, OrderedDict, etc.), the return type will be dict.

    Args:
        tensors: Nested torch tensors.

    Returns:
        The detached tensors.
    """

    import torch
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(torch_nested_detach(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return {k: torch_nested_detach(t) for k, t in tensors.items()}
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    return tensors
