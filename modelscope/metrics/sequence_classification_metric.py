# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
    """The metric computation class for sequence classification tasks.

    This metric class calculates accuracy/F1 of all the input batches.

    Args:
        label_name: The key of label column in the 'inputs' arg.
        logit_name: The key of logits column in the 'inputs' arg.
    """

    def __init__(self,
                 label_name=OutputKeys.LABELS,
                 logit_name=OutputKeys.LOGITS,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []
        self.label_name = label_name
        self.logit_name = logit_name

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = inputs[self.label_name]
        eval_results = outputs[self.logit_name]
        self.preds.append(
            torch_nested_numpify(torch_nested_detach(eval_results)))
        self.labels.append(
            torch_nested_numpify(torch_nested_detach(ground_truths)))

    def evaluate(self):
        preds = np.concatenate(self.preds, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        assert len(preds.shape) == 2, 'Only support predictions with shape: (batch_size, num_labels),' \
                                      'multi-label classification is not supported in this metric class.'
        preds_max = np.argmax(preds, axis=1)
        if preds.shape[1] > 2:
            metrics = {
                MetricKeys.ACCURACY: accuracy_score(labels, preds_max),
                MetricKeys.Micro_F1:
                f1_score(labels, preds_max, average='micro'),
                MetricKeys.Macro_F1:
                f1_score(labels, preds_max, average='macro'),
            }

            metrics[MetricKeys.F1] = metrics[MetricKeys.Micro_F1]
            return metrics
        else:
            metrics = {
                MetricKeys.ACCURACY:
                accuracy_score(labels, preds_max),
                MetricKeys.Binary_F1:
                f1_score(labels, preds_max, average='binary'),
            }
            metrics[MetricKeys.F1] = metrics[MetricKeys.Binary_F1]
            return metrics

    def merge(self, other: 'SequenceClassificationMetric'):
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        return self.preds, self.labels, self.label_name, self.logit_name

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.labels, self.label_name, self.logit_name = state
