# Copyright (c) Alibaba, Inc. and its affiliates.
# Following code is partialy borrowed from openmmlab/mmcv
import functools
import os
import pickle
import random
import numpy as np
import socket
import subprocess
import tempfile
from typing import Callable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from modelscope.utils.nlp import mpu


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

    #torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    #torch.cuda.set_device(local_rank)
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
    #torch.cuda.set_device(proc_id % num_gpus)
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


def initialize_distributed(rank):
    """Initialize torch.distributed."""
    # Manually set the device ids.
    #torch.multiprocessing.set_start_method("spawn")
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '12345')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=8, rank=rank,
        init_method=init_method)
    # Set the model-parallel communicators.
    mpu.initialize_model_parallel(8)


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_master():
    rank, _ = get_dist_info()
    return rank == 0


def master_only(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def create_device(cpu: bool = False) -> torch.DeviceObjType:
    use_cuda = torch.cuda.is_available() and not cpu
    if use_cuda:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    return device


def make_tmp_dir():
    """Make sure each rank has the same temporary directory on the distributed mode.
    """
    rank, world_size = get_dist_info()
    if world_size <= 1:
        return tempfile.mkdtemp()

    tmpdir = None
    if rank == 0:
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
    rank, _ = get_dist_info()
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
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        raise ValueError(f'Random seed should be positive, current seed is {seed}')


def set_random_seed_mpu(seed):
    set_random_seed(seed)
    mpu.model_parallel_cuda_manual_seed(seed)
