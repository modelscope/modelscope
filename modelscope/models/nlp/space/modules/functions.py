# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn.functional as F


def unsqueeze(input, dims):
    """ Implement multi-dimension unsqueeze function. """
    if isinstance(dims, (list, tuple)):
        dims = [
            dim if dim >= 0 else dim + len(input.shape) + 1 for dim in dims
        ]
        dims = sorted(dims, reverse=True)
        shape = list(input.shape)
        for dim in dims:
            shape.insert(dim, 1)
        return torch.reshape(input, shape)
    elif isinstance(dims, int):
        return input.unsqueeze(dims)
    else:
        raise ValueError('Warning: type(dims) must in (list, tuple, int)!')


def gumbel_softmax(input, tau=1, eps=1e-10):
    """ Basic implement of gumbel_softmax. """
    U = torch.tensor(np.random.rand(*input.shape))
    gumbel = 0.0 - torch.log(eps - torch.log(U + eps))
    y = input + gumbel
    return F.softmax(y / tau)


def equal(x, y, dtype=None):
    """ Implement equal in dygraph mode. (paddle) """
    if dtype is None:
        dtype = 'float32'
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    out = np.equal(x, y).astype(dtype)
    return torch.tensor(out)


def not_equal(x, y, dtype=None):
    """ Implement not_equal in dygraph mode. (paddle) """
    return 1 - equal(x, y, dtype)


if __name__ == '__main__':
    a = torch.tensor([[1, 1], [3, 4]])
    b = torch.tensor([[1, 1], [3, 4]])
    c = torch.equal(a, a)
    c1 = equal(a, 3)
    d = 1 - torch.not_equal(a, 3).float()
    print(c)
    print(c1)
    print(d)
    e = F.gumbel_softmax(a)
    f = a.unsqueeze(a)
    g = unsqueeze(a, dims=[0, 0, 1])
    print(g, g.shape)
