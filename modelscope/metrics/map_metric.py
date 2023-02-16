# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.outputs import OutputKeys
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.multi_average_precision)
class AveragePrecisionMetric(Metric):
    """The metric computation class for multi avarage precision classes.

    This metric class calculates multi avarage precision for the whole input batches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []
        self.thresh = kwargs.get('threshold', 0.5)

    def add(self, outputs: Dict, inputs: Dict):
        label_name = OutputKeys.LABEL if OutputKeys.LABEL in inputs else OutputKeys.LABELS
        ground_truths = inputs[label_name]
        eval_results = outputs[label_name]
        for key in [
                OutputKeys.CAPTION, OutputKeys.TEXT, OutputKeys.BOXES,
                OutputKeys.LABELS, OutputKeys.SCORES
        ]:
            if key in outputs and outputs[key] is not None:
                eval_results = outputs[key]
                break
        assert type(ground_truths) == type(eval_results)
        for truth in ground_truths:
            self.labels.append(truth)
        for result in eval_results:
            if isinstance(truth, str):
                self.preds.append(result.strip().replace(' ', ''))
            else:
                self.preds.append(result)

    def evaluate(self):
        assert len(self.preds) == len(self.labels)
        scores = self._calculate_ap_score(self.preds, self.labels, self.thresh)
        return {MetricKeys.mAP: scores.mean().item()}

    def merge(self, other: 'AveragePrecisionMetric'):
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        return self.preds, self.labels, self.thresh

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.labels, self.thresh = state

    def _calculate_ap_score(self, preds, labels, thresh=0.5):
        hyps = np.array(preds)
        refs = np.array(labels)
        a = np.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2])
        b = np.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])
        interacts = np.concatenate([a, b], axis=1)
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (
            hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (
            area_predictions + area_targets - area_interacts + 1e-6)
        return (ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)
