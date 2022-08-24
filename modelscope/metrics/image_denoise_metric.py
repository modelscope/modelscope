from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.image_denoise_metric)
class ImageDenoiseMetric(Metric):
    """The metric computation class for image denoise classes.
    """
    pred_name = 'pred'
    label_name = 'target'

    def __init__(self):
        self.preds = []
        self.labels = []

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = outputs[ImageDenoiseMetric.label_name]
        eval_results = outputs[ImageDenoiseMetric.pred_name]
        self.preds.append(
            torch_nested_numpify(torch_nested_detach(eval_results)))
        self.labels.append(
            torch_nested_numpify(torch_nested_detach(ground_truths)))

    def evaluate(self):
        psnr_list, ssim_list = [], []
        for (pred, label) in zip(self.preds, self.labels):
            psnr_list.append(
                peak_signal_noise_ratio(label[0], pred[0], data_range=255))
            ssim_list.append(
                structural_similarity(
                    label[0], pred[0], multichannel=True, data_range=255))
        return {
            MetricKeys.PSNR: np.mean(psnr_list),
            MetricKeys.SSIM: np.mean(ssim_list)
        }
