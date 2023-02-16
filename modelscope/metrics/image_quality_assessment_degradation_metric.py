# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
import tempfile
from collections import defaultdict
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
    module_name=Metrics.image_quality_assessment_degradation_metric)
class ImageQualityAssessmentDegradationMetric(Metric):
    """The metric for image-quality-assessment-degradation task.
    """

    def __init__(self):
        self.inputs = defaultdict(list)
        self.outputs = defaultdict(list)

    def add(self, outputs: Dict, inputs: Dict):
        item_degradation_id = outputs['item_id'][0] + outputs[
            'distortion_type'][0]
        if outputs['distortion_type'][0] in ['01', '02', '03']:
            pred = outputs['blur_degree']
        elif outputs['distortion_type'][0] in ['09', '10', '21']:
            pred = outputs['comp_degree']
        elif outputs['distortion_type'][0] in ['11', '12', '13', '14']:
            pred = outputs['noise_degree']
        else:
            return

        self.outputs[item_degradation_id].append(pred[0].float())
        self.inputs[item_degradation_id].append(outputs['target'].float())

    def evaluate(self):
        degree_plccs = []
        degree_sroccs = []

        for item_degradation_id, degree_value in self.inputs.items():
            degree_label = torch.cat(degree_value).flatten().data.cpu().numpy()
            degree_pred = torch.cat(self.outputs[item_degradation_id]).flatten(
            ).data.cpu().numpy()
            degree_plcc = pearsonr(degree_label, degree_pred)[0]
            degree_srocc = spearmanr(degree_label, degree_pred)[0]
            degree_plccs.append(degree_plcc)
            degree_sroccs.append(degree_srocc)
        degree_plcc_mean = np.array(degree_plccs).mean()
        degree_srocc_mean = np.array(degree_sroccs).mean()

        return {
            MetricKeys.PLCC: degree_plcc_mean,
            MetricKeys.SRCC: degree_srocc_mean,
        }

    def merge(self, other: 'ImageQualityAssessmentDegradationMetric'):
        self.inputs.extend(other.inputs)
        self.outputs.extend(other.outputs)

    def __getstate__(self):
        return self.inputs, self.outputs

    def __setstate__(self, state):
        self.inputs, self.outputs = state
