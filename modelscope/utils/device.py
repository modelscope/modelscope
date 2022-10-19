# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager

from modelscope.utils.constant import Devices, Frameworks
from modelscope.utils.import_utils import is_tf_available, is_torch_available
from modelscope.utils.logger import get_logger

logger = get_logger()


def verify_device(device_name):
    """ Verify device is valid, device should be either cpu, cuda, gpu, cuda:X or gpu:X.

    Args:
        device (str):  device str, should be either cpu, cuda, gpu, gpu:X or cuda:X
            where X is the ordinal for gpu device.

    Return:
        device info (tuple):  device_type and device_id, if device_id is not set, will use 0 as default.
    """
    err_msg = 'device should be either cpu, cuda, gpu, gpu:X or cuda:X where X is the ordinal for gpu device.'
    assert device_name is not None and device_name != '', err_msg
    device_name = device_name.lower()
    eles = device_name.split(':')
    assert len(eles) <= 2, err_msg
    assert device_name is not None
    assert eles[0] in ['cpu', 'cuda', 'gpu'], err_msg
    device_type = eles[0]
    device_id = None
    if len(eles) > 1:
        device_id = int(eles[1])
    if device_type == 'cuda':
        device_type = Devices.gpu
    if device_type == Devices.gpu and device_id is None:
        device_id = 0
    return device_type, device_id


@contextmanager
def device_placement(framework, device_name='gpu:0'):
    """ Device placement function, allow user to specify which device to place model or tensor
    Args:
        framework (str):  tensorflow or pytorch.
        device (str):  gpu or cpu to use, if you want to specify certain gpu,
            use gpu:$gpu_id or cuda:$gpu_id.

    Returns:
        Context manager

    Examples:

    ```python
    # Requests for using model on cuda:0 for gpu
    with device_placement('pytorch', device='gpu:0'):
        model = Model.from_pretrained(...)
    ```
    """
    device_type, device_id = verify_device(device_name)

    if framework == Frameworks.tf:
        import tensorflow as tf
        if device_type == Devices.gpu and not tf.test.is_gpu_available():
            logger.debug(
                'tensorflow: cuda is not available, using cpu instead.')
        device_type = Devices.cpu
        if device_type == Devices.cpu:
            with tf.device('/CPU:0'):
                yield
        else:
            if device_type == Devices.gpu:
                with tf.device(f'/device:gpu:{device_id}'):
                    yield

    elif framework == Frameworks.torch:
        import torch
        if device_type == Devices.gpu:
            if torch.cuda.is_available():
                torch.cuda.set_device(f'cuda:{device_id}')
            else:
                logger.debug(
                    'pytorch: cuda is not available, using cpu instead.')
        yield
    else:
        yield


def create_device(device_name):
    """ create torch device

    Args:
        device_name (str):  cpu, gpu, gpu:0, cuda:0 etc.
    """
    import torch
    device_type, device_id = verify_device(device_name)
    use_cuda = False
    if device_type == Devices.gpu:
        use_cuda = True
        if not torch.cuda.is_available():
            logger.info('cuda is not available, using cpu instead.')
            use_cuda = False

    if use_cuda:
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')

    return device


def get_device():
    import torch
    from torch import distributed as dist
    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized(
        ) and 'LOCAL_RANK' in os.environ:
            device_id = f"cuda:{os.environ['LOCAL_RANK']}"
        else:
            device_id = 'cuda:0'
    else:
        device_id = 'cpu'
    return torch.device(device_id)
