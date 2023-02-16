# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.outputs import OutputKeys
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(group_key=default_group, module_name=Metrics.NED)
class NedMetric(Metric):
    """The ned metric computation class for classification classes.

    This metric class calculates the levenshtein distance between sentences for the whole input batches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []

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
        if isinstance(ground_truths, list):
            self.preds.extend(eval_results)
            self.labels.extend(ground_truths)
        elif isinstance(ground_truths, np.ndarray):
            self.preds.extend(eval_results.tolist())
            self.labels.extend(ground_truths.tolist())
        else:
            raise Exception('only support list or np.ndarray')

    def evaluate(self):
        assert len(self.preds) == len(self.labels)
        return {
            MetricKeys.NED: (np.asarray([
                1.0 - NedMetric._distance(pred, ref)
                for pred, ref in zip(self.preds, self.labels)
            ])).mean().item()
        }

    def merge(self, other: 'NedMetric'):
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        return self.preds, self.labels

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.labels = state

    @staticmethod
    def _distance(pred, ref):
        if pred is None or ref is None:
            raise TypeError('Argument (pred or ref) is NoneType.')
        if pred == ref:
            return 0.0
        if len(pred) == 0:
            return len(ref)
        if len(ref) == 0:
            return len(pred)
        m_len = max(len(pred), len(ref))
        if m_len == 0:
            return 0.0

        def levenshtein(s0, s1):
            v0 = [0] * (len(s1) + 1)
            v1 = [0] * (len(s1) + 1)

            for i in range(len(v0)):
                v0[i] = i

            for i in range(len(s0)):
                v1[0] = i + 1
                for j in range(len(s1)):
                    cost = 1
                    if s0[i] == s1[j]:
                        cost = 0
                    v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
                v0, v1 = v1, v0
            return v0[len(s1)]

        return levenshtein(pred, ref) / m_len
