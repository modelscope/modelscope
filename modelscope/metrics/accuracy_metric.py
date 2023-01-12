# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.outputs import OutputKeys
from modelscope.utils.chinese_utils import remove_space_between_chinese_chars
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(group_key=default_group, module_name=Metrics.accuracy)
class AccuracyMetric(Metric):
    """The metric computation class for classification classes.

    This metric class calculates accuracy for the whole input batches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []

    def add(self, outputs: Dict, inputs: Dict):
        label_name = OutputKeys.LABEL if OutputKeys.LABEL in inputs else OutputKeys.LABELS
        ground_truths = inputs[label_name]
        eval_results = None
        for key in [
                OutputKeys.CAPTION, OutputKeys.TEXT, OutputKeys.BOXES,
                OutputKeys.LABEL, OutputKeys.LABELS, OutputKeys.SCORES
        ]:
            if key in outputs and outputs[key] is not None:
                eval_results = outputs[key]
                break
        assert type(ground_truths) == type(eval_results)
        for truth in ground_truths:
            self.labels.append(truth)
        for result in eval_results:
            if isinstance(truth, str):
                if isinstance(result, list):
                    result = result[0]
                assert isinstance(result, str), 'both truth and pred are str'
                self.preds.append(remove_space_between_chinese_chars(result))
            else:
                self.preds.append(result)

    def evaluate(self):
        assert len(self.preds) == len(self.labels)
        return {
            MetricKeys.ACCURACY: (np.asarray([
                pred == ref for pred, ref in zip(self.preds, self.labels)
            ])).mean().item()
        }

    def merge(self, other: 'AccuracyMetric'):
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        return self.preds, self.labels

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.labels = state
