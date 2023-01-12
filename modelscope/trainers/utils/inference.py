# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import pickle
import shutil
from collections.abc import Mapping

import torch
from torch import distributed as dist
from tqdm import tqdm

from modelscope.utils.data_utils import to_device
from modelscope.utils.torch_utils import (broadcast, get_dist_info, is_master,
                                          make_tmp_dir)


def single_gpu_test(trainer,
                    data_loader,
                    device,
                    metric_classes=None,
                    vis_closure=None,
                    data_loader_iters=None):
    """Test model in EpochBasedTrainer with a single gpu.

    Args:
        trainer (modelscope.trainers.EpochBasedTrainer): Trainer to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        device (str | torch.device): The target device for the data.
        metric_classes (List): List of Metric class that uses to collect metrics.
        vis_closure (Callable): Collect data for TensorboardHook.
        data_loader_iters (int): Used when dataset has no attribute __len__ or only load part of dataset.

    Returns:
        list: The prediction results.
    """
    dataset = data_loader.dataset
    progress_with_iters = False
    if data_loader_iters is None:
        try:
            data_len = len(dataset)
        except Exception as e:
            logging.error(e)
            raise ValueError(
                'Please implement ``__len__`` method for your dataset, or provide ``data_loader_iters``'
            )
        desc = 'Total test samples'
    else:
        progress_with_iters = True
        data_len = data_loader_iters
        desc = 'Test iterations'

    with tqdm(total=data_len, desc=desc) as pbar:
        for i, data in enumerate(data_loader):
            data = to_device(data, device)
            evaluate_batch(trainer, data, metric_classes, vis_closure)

            if progress_with_iters:
                batch_size = 1  # iteration count
            else:
                if isinstance(data, Mapping):
                    if 'nsentences' in data:
                        batch_size = data['nsentences']
                    else:
                        try:
                            batch_size = len(next(iter(data.values())))
                        except Exception:
                            batch_size = data_loader.batch_size
                else:
                    batch_size = len(data)
            for _ in range(batch_size):
                pbar.update()

            if progress_with_iters and (i + 1) >= data_len:
                break

    return get_metric_values(metric_classes)


def multi_gpu_test(trainer,
                   data_loader,
                   device,
                   metric_classes=None,
                   vis_closure=None,
                   tmpdir=None,
                   gpu_collect=False,
                   data_loader_iters_per_gpu=None):
    """Test model in EpochBasedTrainer with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        trainer (modelscope.trainers.EpochBasedTrainer): Trainer to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        device: (str | torch.device): The target device for the data.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        data_loader_iters_per_gpu (int): Used when dataset has no attribute __len__ or only load part of dataset.
    Returns:
        list: The prediction results.
    """
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    progress_with_iters = False
    if data_loader_iters_per_gpu is None:
        try:
            data_len = len(dataset)
            total_samples = data_len
        except Exception as e:
            logging.error(e)
            raise ValueError(
                'Please implement ``__len__`` method for your dataset, or provide ``data_loader_iters_per_gpu``'
            )
        desc = 'Total test samples with multi gpus'
    else:
        total_samples = 0
        progress_with_iters = True
        data_len = data_loader_iters_per_gpu * world_size
        desc = 'Total test iterations with multi gpus'

    count = 0
    with tqdm(total=data_len, desc=desc) as pbar:
        for i, data in enumerate(data_loader):
            data = to_device(data, device)

            evaluate_batch(trainer, data, metric_classes, vis_closure)

            if isinstance(data, Mapping):
                if 'nsentences' in data:
                    batch_size = data['nsentences']
                else:
                    batch_size = len(next(iter(data.values())))
            else:
                batch_size = len(data)
            if i >= (data_len // world_size) - 1:
                total_samples = torch.LongTensor([batch_size
                                                  ]).to(trainer.model.device)
                dist.all_reduce(total_samples, op=dist.reduce_op.SUM)
                total_samples = total_samples.item()
            else:
                total_samples = batch_size * world_size
            if progress_with_iters:
                iter_cnt_all = world_size
            else:
                iter_cnt_all = total_samples
                count += iter_cnt_all

            if rank == 0:
                if count > data_len:
                    iter_cnt_all = data_len - (count - iter_cnt_all)
                for _ in range(iter_cnt_all):
                    pbar.update()

            if progress_with_iters and (i + 1) >= data_len:
                break

    # collect results and data from all ranks
    if gpu_collect:
        metric_classes_list = collect_results_gpu(metric_classes)
    else:
        if tmpdir is None:
            tmpdir = make_tmp_dir()
        metric_classes_list = collect_results_cpu(
            metric_classes, os.path.join(tmpdir, 'metrics'))

    metric_classes = merge_metrics(metric_classes_list)

    return get_metric_values(metric_classes)


def evaluate_batch(trainer, data, metric_classes, vis_closure):
    batch_result = trainer.evaluation_step(data)

    if metric_classes is not None:
        for metric_cls in metric_classes:
            metric_cls.add(batch_result, data)

    if vis_closure is not None:
        # trainer.visualization
        vis_closure(batch_result)


def get_metric_values(metric_classes):
    rank, world_size = get_dist_info()
    metric_values = {}
    if rank == 0:
        for metric_cls in metric_classes:
            metric_values.update(metric_cls.evaluate())
    if world_size > 1:
        metric_values = broadcast(metric_values, 0)
    return metric_values


def collect_results_cpu(result_part, tmpdir=None):
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

        # remove tmp dir
        shutil.rmtree(tmpdir)
        return part_list


def collect_results_gpu(result_part):
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

        return part_list


def merge_metrics(metric_classes_list):
    if metric_classes_list is None:
        return None

    metric_classes_0 = metric_classes_list[0]
    for metric_classes_i in metric_classes_list[1:]:
        for cls_0, cls_i in zip(metric_classes_0, metric_classes_i):
            cls_0.merge(cls_i)
    return metric_classes_0
