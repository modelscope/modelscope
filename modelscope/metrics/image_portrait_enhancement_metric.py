# Part of the implementation is borrowed and modified from BasicSR, publicly available at
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py
from typing import Dict

import cv2
import numpy as np

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


def calculate_psnr(img, img2):
    assert img.shape == img2.shape, (
        f'Image shapes are different: {img.shape}, {img2.shape}.')

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


@METRICS.register_module(
    group_key=default_group,
    module_name=Metrics.image_portrait_enhancement_metric)
class ImagePortraitEnhancementMetric(Metric):
    """The metric for image-portrait-enhancement task.
    """

    def __init__(self):
        self.preds = []
        self.targets = []

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = outputs['target']
        eval_results = outputs['pred']

        self.preds.extend(eval_results)
        self.targets.extend(ground_truths)

    def evaluate(self):
        psnrs = [
            calculate_psnr(pred, target)
            for pred, target in zip(self.preds, self.targets)
        ]

        return {MetricKeys.PSNR: sum(psnrs) / len(psnrs)}

    def merge(self, other: 'ImagePortraitEnhancementMetric'):
        self.preds.extend(other.preds)
        self.targets.extend(other.targets)

    def __getstate__(self):
        return self.preds, self.targets

    def __setstate__(self, state):
        self.preds, self.targets = state
