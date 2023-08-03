# Copyright (c) Alibaba, Inc. and its affiliates.
# Following code is partialy borrowed from openmmlab/mmcv
import functools
import os
import pickle
import random
import socket
import subprocess
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from packaging import version
from torch import distributed as dist


def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port: int) -> bool:
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def compile_model(model, **compile_options):
    # Compile the model with torch 2.0
    if hasattr(model, 'compile'):
        model = model.compile(**compile_options)
    elif version.parse(torch.__version__) >= version.parse('2.0.0.dev'):
        model = torch.compile(model, **compile_options)
    else:
        print(
            'Compiling model needs torch version > 2.0.0, '
            f'your torch version is: {torch.__version__}, origin model will be returned.'
        )
    return model


def init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend: str, **kwargs) -> None:
    # rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend: str, port: Optional[int] = None) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info(group=None) -> Tuple[int, int]:
    """Get dist info of a specified group

    Args:
        group: The parallel group, default None, for the global group

    Returns:
        A tuple of the current rank and world_size of the group
    """
    if is_dist():
        from modelscope.utils.megatron_utils import is_megatron_initialized
        if group is None and is_megatron_initialized():
            from megatron_util import mpu
            group = mpu.get_data_parallel_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """
    Helper function to synchronize (barrier)
    among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master(group=None):
    return dist.get_rank(group) == 0 if is_dist() else True


def master_only(group=None):

    def decorate(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_master(group):
                return func(*args, **kwargs)

        return wrapper

    return decorate


def make_tmp_dir():
    """Make sure each rank has the same temporary directory on the distributed mode.
    """
    if not is_dist():
        return tempfile.mkdtemp()

    tmpdir = None
    if is_master():
        tmpdir = tempfile.mkdtemp()

    dist.barrier()
    tmpdir = broadcast(tmpdir, 0)

    return tmpdir


def broadcast(inputs, src):
    """
    Broadcasts the inputs to all ranks.

    Arguments:
        inputs : Any objects that can be serialized by pickle.
        src (int): Source rank.
    Returns:
        Each rank returns the same value as src.
    """
    rank = dist.get_rank()
    shape_tensor = torch.tensor([0], device='cuda')

    if rank == src:
        inputs_tensor = torch.tensor(
            bytearray(pickle.dumps(inputs)), dtype=torch.uint8, device='cuda')
        shape_tensor = torch.tensor(inputs_tensor.shape, device='cuda')

    dist.barrier()
    dist.broadcast(shape_tensor, src)

    if rank != src:
        inputs_tensor = torch.full((shape_tensor.item(), ),
                                   0,
                                   dtype=torch.uint8,
                                   device='cuda')

    dist.barrier()
    dist.broadcast(inputs_tensor, src)

    return pickle.loads(inputs_tensor.cpu().numpy().tobytes())


def set_random_seed(seed):
    if seed is not None and seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        raise ValueError(
            f'Random seed should be positive, current seed is {seed}')


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['gloo', 'nccl']
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger.warning(
            'Rank {} trying to all-gather {:.2f} GB of data on device {}'.
            format(get_rank(),
                   len(buffer) / (1024**3), device))
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), 'comm.gather/all_gather must be called from ranks within the group!'
    local_size = torch.tensor([tensor.numel()],
                              dtype=torch.int64,
                              device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size, ),
                              dtype=torch.uint8,
                              device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size, ), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_on_same_device(model: torch.nn.Module) -> bool:
    device_set = set(str(p.device) for p in model.parameters()) - {'cpu'}
    return len(device_set) <= 1
