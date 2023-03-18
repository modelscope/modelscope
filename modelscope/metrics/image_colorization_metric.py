# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from modelscope.metainfo import Metrics
from modelscope.models.cv.image_inpainting.modules.inception import InceptionV3
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)
from .base import Metric
from .builder import METRICS, MetricKeys
from .image_denoise_metric import calculate_psnr
from .image_inpainting_metric import FIDScore


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.image_colorization_metric)
class ImageColorizationMetric(Metric):
    """The metric computation class for image colorization.
    """

    def __init__(self):
        self.preds = []
        self.targets = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.FID = FIDScore().to(device)

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = outputs['preds']
        eval_results = outputs['targets']
        self.preds.append(eval_results)
        self.targets.append(ground_truths)

    def evaluate(self):
        psnr_list = []
        for (pred, target) in zip(self.preds, self.targets):
            self.FID(pred, target)
            psnr_list.append(calculate_psnr(target[0], pred[0], crop_border=0))
        fid = self.FID.get_value()
        return {MetricKeys.PSNR: np.mean(psnr_list), MetricKeys.FID: fid}

    def merge(self, other: 'ImageColorizationMetric'):
        self.preds.extend(other.preds)
        self.targets.extend(other.targets)

    def __getstate__(self):
        return self.preds, self.targets

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.targets = state
