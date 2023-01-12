# Copyright (c) Alibaba, Inc. and its affiliates.
from collections.abc import Mapping

import torch

from modelscope.outputs import ModelOutputBase


def to_device(batch, device, non_blocking=False):
    """Put the data to the target cuda device just before the forward function.
    Args:
        batch: The batch data out of the dataloader.
        device: (str | torch.device): The target device for the data.

    Returns: The data to the target device.

    """
    if isinstance(batch, ModelOutputBase):
        for idx in range(len(batch)):
            batch[idx] = to_device(batch[idx], device)
        return batch
    elif isinstance(batch, dict) or isinstance(batch, Mapping):
        return type(batch)({k: to_device(v, device) for k, v in batch.items()})
    elif isinstance(batch, (tuple, list)):
        return type(batch)(to_device(v, device) for v in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    else:
        return batch
