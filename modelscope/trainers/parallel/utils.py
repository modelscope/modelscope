# Copyright (c) Alibaba, Inc. and its affiliates.
from .builder import PARALLEL


def is_parallel(module):
    """Check if a module is wrapped by parallel object.

    The following modules are regarded as parallel object:
     - torch.nn.parallel.DataParallel
     - torch.nn.parallel.distributed.DistributedDataParallel
    You may add you own parallel object by registering it to `modelscope.parallel.PARALLEL`.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the is wrapped by parallel object.
    """
    module_wrappers = []
    for group, module_dict in PARALLEL.modules.items():
        module_wrappers.extend(list(module_dict.values()))

    return isinstance(module, tuple(module_wrappers))
