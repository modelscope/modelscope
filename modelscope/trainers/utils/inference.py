# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
import shutil
import time
from collections.abc import Mapping

import torch
from torch import distributed as dist
from tqdm import tqdm

from modelscope.utils.data_utils import to_device
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.torch_utils import (broadcast, get_dist_info, is_master,
                                          make_tmp_dir)


def single_gpu_test(model, data_loader, device, metric_classes=None):
    """Test model with a single gpu.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        device: (str | torch.device): The target device for the data.
        metric_classes(List): List of Metric class that uses to collect metrics

    Returns:
        list: The prediction results.
    """
    model.eval()
    dataset = data_loader.dataset
    with tqdm(total=len(dataset), desc='test samples') as pbar:
        for data in data_loader:
            data = to_device(data, device)
            with torch.no_grad():
                if isinstance(data, Mapping) and not func_receive_dict_inputs(
                        model.forward):
                    result = model.forward(**data)
                else:
                    result = model.forward(data)
            if metric_classes is not None:
                for metric_cls in metric_classes:
                    metric_cls.add(result, data)

            if isinstance(data, dict):
                batch_size = len(next(iter(data.values())))
            else:
                batch_size = len(data)
            for _ in range(batch_size):
                pbar.update()

    metric_values = {}
    for metric_cls in metric_classes:
        metric_values.update(metric_cls.evaluate())

    return metric_values


def multi_gpu_test(model,
                   data_loader,
                   device,
                   tmpdir=None,
                   gpu_collect=False,
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
        device: (str | torch.device): The target device for the data.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        metric_classes(List): List of Metric class that uses to collect metrics

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    data_list = []
    dataset = data_loader.dataset

    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    rank, world_size = get_dist_info()

    count = 0
    with tqdm(total=len(dataset), desc='test samples with multi gpus') as pbar:
        for _, data in enumerate(data_loader):
            data = to_device(data, device)
            data_list.append(data)
            with torch.no_grad():
                if isinstance(data, Mapping) and not func_receive_dict_inputs(
                        model.forward):
                    result = model.forward(**data)
                else:
                    result = model.forward(data)
            results.append(result)

            if rank == 0:
                if isinstance(data, dict):
                    batch_size = len(next(iter(data.values())))
                else:
                    batch_size = len(data)
                batch_size_all = batch_size * world_size
                count += batch_size_all
                if count > len(dataset):
                    batch_size_all = len(dataset) - (count - batch_size_all)
                for _ in range(batch_size_all):
                    pbar.update()

    # TODO: allgather data list may cost a lot of memory and needs to be redesigned
    # collect results and data from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
        data_list = collect_results_gpu(data_list, len(dataset))
    else:
        if tmpdir is None:
            tmpdir = make_tmp_dir()
        results = collect_results_cpu(results, len(dataset),
                                      os.path.join(tmpdir, 'predict'))
        data_list = collect_results_cpu(data_list, len(dataset),
                                        os.path.join(tmpdir, 'groundtruth'))

    if is_master():
        assert len(data_list) == len(
            results), f'size mismatch {len(data_list)} and {len(results)}'
        if metric_classes is not None:
            for i in range(len(data_list)):
                for metric_cls in metric_classes:
                    metric_cls.add(results[i], data_list[i])

    metric_values = {}
    if rank == 0:
        for metric_cls in metric_classes:
            metric_values.update(metric_cls.evaluate())
    if world_size > 1:
        metric_values = broadcast(metric_values, 0)

    return metric_values


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
    if tmpdir is None:
        tmpdir = make_tmp_dir()
    if not os.path.exists(tmpdir) and is_master():
        os.makedirs(tmpdir)
    dist.barrier()

    # dump the part result to the dir
    with open(os.path.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:
        pickle.dump(result_part, f)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            with open(part_file, 'rb') as f:
                part_result = pickle.load(f)
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
