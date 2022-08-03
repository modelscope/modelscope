from typing import Dict, List, Union

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.outputs import OutputKeys
from modelscope.utils.registry import default_group
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.seq_cls_metric)
class SequenceClassificationMetric(Metric):
    """The metric computation class for sequence classification classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []

    def add(self, outputs: Dict, inputs: Dict):
        label_name = OutputKeys.LABEL if OutputKeys.LABEL in inputs else OutputKeys.LABELS
        ground_truths = inputs[label_name]
        eval_results = outputs[OutputKeys.LOGITS]
        self.preds.append(
            torch_nested_numpify(torch_nested_detach(eval_results)))
        self.labels.append(
            torch_nested_numpify(torch_nested_detach(ground_truths)))

    def evaluate(self):
        preds = np.concatenate(self.preds, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        preds = np.argmax(preds, axis=1)
        return {
            MetricKeys.ACCURACY:
            (preds == labels).astype(np.float32).mean().item()
        }
