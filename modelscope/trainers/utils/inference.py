# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
import shutil
import tempfile
import time

import torch
from torch import distributed as dist
from tqdm import tqdm

from modelscope.utils.torch_utils import get_dist_info


def single_gpu_test(model,
                    data_loader,
                    data_collate_fn=None,
                    metric_classes=None):
    """Test model with a single gpu.

    Args:
        data_collate_fn: An optional data_collate_fn before fed into the model
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        metric_classes(List): List of Metric class that uses to collect metrics

    Returns:
        list: The prediction results.
    """
    model.eval()
    dataset = data_loader.dataset
    with tqdm(total=len(dataset), desc='test samples') as pbar:
        for data in data_loader:
            if data_collate_fn is not None:
                data = data_collate_fn(data)
            with torch.no_grad():
                result = model(**data)
            if metric_classes is not None:
                for metric_cls in metric_classes:
                    metric_cls.add(result, data)

            batch_size = len(result)
            for _ in range(batch_size):
                pbar.update()


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   data_collate_fn=None,
                   metric_classes=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        data_collate_fn: An optional data_collate_fn before fed into the model
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        metric_classes(List): List of Metric class that uses to collect metrics

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset

    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    count = 0
    with tqdm(total=len(dataset), desc='test samples with multi gpus') as pbar:
        for _, data in enumerate(data_loader):
            if data_collate_fn is not None:
                data = data_collate_fn(data)
            with torch.no_grad():
                result = model(**data)
            results.extend(result)

            rank, world_size = get_dist_info()
            if rank == 0:
                batch_size = len(result)
                batch_size_all = batch_size * world_size
                count += batch_size_all
                if count > len(dataset):
                    batch_size_all = len(dataset) - (count - batch_size_all)
                for _ in range(batch_size_all):
                    pbar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    ground_truths = [dataset[i] for i in range(len(dataset))]
    if metric_classes is not None:
        for metric_cls in metric_classes:
            metric_cls.add(results, ground_truths)


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # TODO create a random tmp dir if it is not specified
    if tmpdir is None:
        tmpdir = tempfile.gettempdir()
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    # dump the part result to the dir
    pickle.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_result = pickle.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
