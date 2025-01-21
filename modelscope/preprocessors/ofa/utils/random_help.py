# Copyright (c) Alibaba, Inc. and its affiliates.

import torch

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def get_rng_state():
    r"""
    Get random number generator state of torch, xla and cuda.
    """
    state = {'torch_rng_state': torch.get_rng_state()}
    if xm is not None:
        state['xla_rng_state'] = xm.get_rng_state()
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    r"""
    Set random number generator state of torch, xla and cuda.
    """
    torch.set_rng_state(state['torch_rng_state'])
    if xm is not None:
        xm.set_rng_state(state['xla_rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda_rng_state'])


class set_torch_seed(object):
    r"""
    Set random seed to torch, xla and cuda.
    """

    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if xm is not None:
            xm.set_rng_state(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)
