# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/facebookresearch/detectron2/blob/v0.3/detectron2/evaluation/evaluator.py

import datetime
import logging
import time

import torch
from detectron2.evaluation.evaluator import inference_context

from ..models.calibration_layer import PrototypicalCalibrationBlock


def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size(
    ) if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info('Start initializing PCB module, please wait a seconds...')
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info('Start inference on {} images'.format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup)
                                - duration))
                logger.info(
                    'Inference done {}/{}. {:.4f} s / img. ETA={}'.format(
                        idx + 1, total, seconds_per_img, str(eta)))

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        'Total inference time: {} ({:.6f} s / img per device, on {} devices)'.
        format(total_time_str, total_time / (total - num_warmup), num_devices))
    total_compute_time_str = str(
        datetime.timedelta(seconds=int(total_compute_time)))
    time_per_device = total_compute_time / (total - num_warmup)
    logger.info(
        'Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)'
        .format(total_compute_time_str, time_per_device, num_devices))

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
