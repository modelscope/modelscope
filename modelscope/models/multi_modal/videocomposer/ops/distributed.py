# Copyright (c) Alibaba, Inc. and its affiliates.

import functools
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function

__all__ = [
    'is_dist_initialized', 'get_world_size', 'get_rank', 'new_group',
    'destroy_process_group', 'barrier', 'broadcast', 'all_reduce', 'reduce',
    'gather', 'all_gather', 'reduce_dict', 'get_global_gloo_group',
    'generalized_all_gather', 'generalized_gather', 'scatter',
    'reduce_scatter', 'send', 'recv', 'isend', 'irecv', 'shared_random_seed',
    'diff_all_gather', 'diff_all_reduce', 'diff_scatter', 'diff_copy',
    'spherical_kmeans', 'sinkhorn'
]


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size(group=None):
    return dist.get_world_size(group) if is_dist_initialized() else 1


def get_rank(group=None):
    return dist.get_rank(group) if is_dist_initialized() else 0


def new_group(ranks=None, **kwargs):
    if is_dist_initialized():
        return dist.new_group(ranks, **kwargs)
    return None


def destroy_process_group():
    if is_dist_initialized():
        dist.destroy_process_group()


def barrier(group=None, **kwargs):
    if get_world_size(group) > 1:
        dist.barrier(group, **kwargs)


def broadcast(tensor, src, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.broadcast(tensor, src, group, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.all_reduce(tensor, op, group, **kwargs)


def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.reduce(tensor, dst, op, group, **kwargs)


def gather(tensor, dst=0, group=None, **kwargs):
    rank = get_rank()
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor)
                   for _ in range(world_size)] if rank == dst else None
    dist.gather(tensor, tensor_list, dst, group, **kwargs)
    return tensor_list


def all_gather(tensor, uniform_size=True, group=None, **kwargs):
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    assert tensor.is_contiguous(
    ), 'ops.all_gather requires the tensor to be contiguous()'

    if uniform_size:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)
        return tensor_list
    else:
        # collect tensor shapes across GPUs
        shape = tuple(tensor.shape)
        shape_list = generalized_all_gather(shape, group)

        # flatten the tensor
        tensor = tensor.reshape(-1)
        size = int(np.prod(shape))
        size_list = [int(np.prod(u)) for u in shape_list]
        max_size = max(size_list)

        # pad to maximum size
        if size != max_size:
            padding = tensor.new_zeros(max_size - size)
            tensor = torch.cat([tensor, padding], dim=0)

        # all_gather
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)

        # reshape tensors
        tensor_list = [
            t[:n].view(s)
            for t, n, s in zip(tensor_list, size_list, shape_list)
        ]
        return tensor_list


@torch.no_grad()
def reduce_dict(input_dict, group=None, reduction='mean', **kwargs):
    assert reduction in ['mean', 'sum']
    world_size = get_world_size(group)
    if world_size == 1:
        return input_dict

    # ensure that the orders of keys are consistent across processes
    if isinstance(input_dict, OrderedDict):
        keys = list(input_dict.keys)
    else:
        keys = sorted(input_dict.keys())
    vals = [input_dict[key] for key in keys]
    vals = torch.stack(vals, dim=0)
    dist.reduce(vals, dst=0, group=group, **kwargs)
    if dist.get_rank(group) == 0 and reduction == 'mean':
        vals /= world_size
    dist.broadcast(vals, src=0, group=group, **kwargs)
    reduced_dict = type(input_dict)([(key, val)
                                     for key, val in zip(keys, vals)])
    return reduced_dict


@functools.lru_cache()
def get_global_gloo_group():
    backend = dist.get_backend()
    assert backend in ['gloo', 'nccl']
    if backend == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['gloo', 'nccl']
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger = logging.getLogger(__name__)
        logger.warning(
            'Rank {} trying to all-gather {:.2f} GB of data on device'
            '{}'.format(get_rank(),
                        len(buffer) / (1024**3), device))
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, \
        'gather/all_gather must be called from ranks within' \
        'the give group!'
    local_size = torch.tensor([tensor.numel()],
                              dtype=torch.int64,
                              device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]

    # gather tensors and compute the maximum size
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # pad tensors to the same size
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size, ),
                              dtype=torch.uint8,
                              device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def generalized_all_gather(data, group=None):
    if get_world_size(group) == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving tensors from all ranks
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


def generalized_gather(data, dst=0, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()
    rank = dist.get_rank()

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving tensors from all ranks to dst
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size, ), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def scatter(data, scatter_list=None, src=0, group=None, **kwargs):
    r"""NOTE: only supports CPU tensor communication.
    """
    if get_world_size(group) > 1:
        return dist.scatter(data, scatter_list, src, group, **kwargs)


def reduce_scatter(output,
                   input_list,
                   op=dist.ReduceOp.SUM,
                   group=None,
                   **kwargs):
    if get_world_size(group) > 1:
        return dist.reduce_scatter(output, input_list, op, group, **kwargs)


def send(tensor, dst, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(
        ), 'ops.send requires the tensor to be contiguous()'
        return dist.send(tensor, dst, group, **kwargs)


def recv(tensor, src=None, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(
        ), 'ops.recv requires the tensor to be contiguous()'
        return dist.recv(tensor, src, group, **kwargs)


def isend(tensor, dst, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(
        ), 'ops.isend requires the tensor to be contiguous()'
        return dist.isend(tensor, dst, group, **kwargs)


def irecv(tensor, src=None, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(
        ), 'ops.irecv requires the tensor to be contiguous()'
        return dist.irecv(tensor, src, group, **kwargs)


def shared_random_seed(group=None):
    seed = np.random.randint(2**31)
    all_seeds = generalized_all_gather(seed, group)
    return all_seeds[0]


def _all_gather(x):
    if not (dist.is_available()
            and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensors = [torch.empty_like(x) for _ in range(world_size)]
    tensors[rank] = x
    dist.all_gather(tensors, x)
    return torch.cat(tensors, dim=0).contiguous()


def _all_reduce(x):
    if not (dist.is_available()
            and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    dist.all_reduce(x)
    return x


def _split(x):
    if not (dist.is_available()
            and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return x.chunk(world_size, dim=0)[rank].contiguous()


class DiffAllGather(Function):
    r"""Differentiable all-gather.
    """

    @staticmethod
    def symbolic(graph, input):
        return _all_gather(input)

    @staticmethod
    def forward(ctx, input):
        return _all_gather(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


class DiffAllReduce(Function):
    r"""Differentiable all-reducd.
    """

    @staticmethod
    def symbolic(graph, input):
        return _all_reduce(input)

    @staticmethod
    def forward(ctx, input):
        return _all_reduce(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DiffScatter(Function):
    r"""Differentiable scatter.
    """

    @staticmethod
    def symbolic(ctx, input):
        return _split(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_gather(grad_output)


class DiffCopy(Function):
    r"""Differentiable copy that reduces all gradients during backward.
    """

    @staticmethod
    def symbolic(graph, input):
        return input

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output)


diff_all_gather = DiffAllGather.apply
diff_all_reduce = DiffAllReduce.apply
diff_scatter = DiffScatter.apply
diff_copy = DiffCopy.apply


@torch.no_grad()
def spherical_kmeans(feats, num_clusters, num_iters=10):
    k, n, c = num_clusters, *feats.size()
    ones = feats.new_ones(n, dtype=torch.long)

    # distributed settings
    world_size = get_world_size()

    # init clusters
    rand_inds = torch.randperm(n)[:int(np.ceil(k / world_size))]
    clusters = torch.cat(all_gather(feats[rand_inds]), dim=0)[:k]

    # variables
    new_clusters = feats.new_zeros(k, c)
    counts = feats.new_zeros(k, dtype=torch.long)

    # iterative Expectation-Maximization
    for step in range(num_iters + 1):
        # Expectation step
        simmat = torch.mm(feats, clusters.t())
        scores, assigns = simmat.max(dim=1)
        if step == num_iters:
            break

        # Maximization step
        new_clusters.zero_().scatter_add_(0,
                                          assigns.unsqueeze(1).repeat(1, c),
                                          feats)
        all_reduce(new_clusters)

        counts.zero_()
        counts.index_add_(0, assigns, ones)
        all_reduce(counts)

        mask = (counts > 0)
        clusters[mask] = new_clusters[mask] / counts[mask].view(-1, 1)
        clusters = F.normalize(clusters, p=2, dim=1)
    return clusters, assigns, scores


@torch.no_grad()
def sinkhorn(Q, eps=0.5, num_iters=3):
    # normalize Q
    Q = torch.exp(Q / eps).t()
    sum_Q = Q.sum()
    all_reduce(sum_Q)
    Q /= sum_Q

    # variables
    n, m = Q.size()
    u = Q.new_zeros(n)
    r = Q.new_ones(n) / n
    c = Q.new_ones(m) / (m * get_world_size())

    # iterative update
    cur_sum = Q.sum(dim=1)
    all_reduce(cur_sum)
    for i in range(num_iters):
        u = cur_sum
        Q *= (r / u).unsqueeze(1)
        Q *= (c / Q.sum(dim=0)).unsqueeze(0)
        cur_sum = Q.sum(dim=1)
        all_reduce(cur_sum)
    return (Q / Q.sum(dim=0, keepdim=True)).t().float()
