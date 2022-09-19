# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.


def torch_nested_numpify(tensors):
    import torch
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(torch_nested_numpify(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        t = tensors.cpu()
        return t.numpy()
    return tensors


def torch_nested_detach(tensors):
    import torch
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(torch_nested_detach(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    return tensors
