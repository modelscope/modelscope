# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.metrics.video_super_resolution_metric.niqe import \
    calculate_niqe
from modelscope.utils.registry import default_group


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.video_super_resolution_metric)
class VideoSuperResolutionMetric(Metric):
    """The metric computation class for real-world video super-resolution classes.
    """
    pred_name = 'pred'

    def __init__(self):
        super(VideoSuperResolutionMetric, self).__init__()
        self.preds = []

    def add(self, outputs: Dict, inputs: Dict):
        eval_results = outputs[VideoSuperResolutionMetric.pred_name]
        self.preds.append(eval_results)

    def evaluate(self):
        niqe_list = []
        for pred in self.preds:
            if isinstance(pred, list):
                for item in pred:
                    niqe_list.append(
                        calculate_niqe(
                            item[0].permute(1, 2, 0).numpy() * 255,
                            crop_border=0))
            else:
                niqe_list.append(
                    calculate_niqe(
                        pred[0].permute(1, 2, 0).numpy() * 255, crop_border=0))
        return {MetricKeys.NIQE: np.mean(niqe_list)}

    def merge(self, other: 'VideoSuperResolutionMetric'):
        self.preds.extend(other.preds)

    def __getstate__(self):
        return self.preds

    def __setstate__(self, state):
        self.__init__()
        self.preds = state
