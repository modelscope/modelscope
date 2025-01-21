# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
from tqdm import tqdm

from modelscope.msdatasets.dataset_cls.custom_datasets.damoyolo import evaluate
from modelscope.utils.logger import get_logger
from modelscope.utils.timer import Timer, get_time_str
from modelscope.utils.torch_utils import (all_gather, get_world_size,
                                          is_master, synchronize)

logger = get_logger()


def compute_on_dataset(model, data_loader, device, timer=None, tta=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device('cpu')
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
                output = model(images.to(device))
            if timer:
                # torch.cuda.synchronize()  consume much time
                timer.toc()
            output = [o.to(cpu_device) if o is not None else o for o in output]
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, output)})
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu,
                                               multi_gpu_infer):
    if multi_gpu_infer:
        all_predictions = all_gather(predictions_per_gpu)
    else:
        all_predictions = [predictions_per_gpu]
    if not is_master():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger.warning(
            'Number of images that were gathered from multiple processes is'
            'not a contiguous set. Some images might be missing from the'
            'evaluation')

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
    model,
    data_loader,
    iou_types=('bbox', ),
    box_only=False,
    device='cuda',
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    multi_gpu_infer=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info('Start evaluation ({} images).'.format(len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device,
                                     inference_timer)
    # wait for all processes to complete before measuring the time
    if multi_gpu_infer:
        synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        'Total run time: {} ({} s / img per device, on {} devices)'.format(
            total_time_str, total_time * num_devices / len(dataset),
            num_devices))
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        'Model inference time: {} ({} s / img per device, on {} devices)'.
        format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        ))

    predictions = _accumulate_predictions_from_multiple_gpus(
        predictions, multi_gpu_infer)
    if not is_master():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        **extra_args)
