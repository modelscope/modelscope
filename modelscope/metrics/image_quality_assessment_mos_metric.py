# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
import tempfile
from typing import Dict

import cv2
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group,
    module_name=Metrics.image_quality_assessment_mos_metric)
class ImageQualityAssessmentMosMetric(Metric):
    """The metric for image-quality-assessment-mos task.
    """

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, outputs: Dict, inputs: Dict):
        self.outputs.append(outputs['pred'].float())
        self.inputs.append(outputs['target'].float())

    def evaluate(self):

        mos_labels = torch.cat(self.inputs).flatten().data.cpu().numpy()
        mos_preds = torch.cat(self.outputs).flatten().data.cpu().numpy()
        mos_plcc = pearsonr(mos_labels, mos_preds)[0]
        mos_srocc = spearmanr(mos_labels, mos_preds)[0]
        mos_rmse = np.sqrt(np.mean((mos_labels - mos_preds)**2))

        return {
            MetricKeys.PLCC: mos_plcc,
            MetricKeys.SRCC: mos_srocc,
            MetricKeys.RMSE: mos_rmse,
        }

    def merge(self, other: 'ImageQualityAssessmentMosMetric'):
        self.inputs.extend(other.inputs)
        self.outputs.extend(other.outputs)

    def __getstate__(self):
        return self.inputs, self.outputs

    def __setstate__(self, state):
        self.inputs, self.outputs = state
