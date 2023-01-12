# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

from modelscope.metainfo import Metrics
from modelscope.outputs import OutputKeys
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(group_key=default_group, module_name=Metrics.PPL)
class PplMetric(Metric):
    """The metric computation class for any classes.

    This metric class calculates perplexity for the whole input batches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_loss: float = 0.
        self.batch_num: int = 0

    def add(self, outputs: Dict, inputs: Dict):
        logits = outputs[OutputKeys.LOGITS]
        labels = inputs[OutputKeys.LABELS]

        in_loss = self._get_loss(logits, labels)
        in_batch_num = self._get_batch_num(inputs[OutputKeys.LABELS])

        self.avg_loss = self._average_loss(in_loss, in_batch_num)
        self.batch_num += in_batch_num

    @staticmethod
    def _get_loss(logits: torch.Tensor, labels: torch.Tensor) -> float:
        labels = labels.view(-1)
        logits = logits.view(labels.shape[0], -1)
        return F.cross_entropy(logits, labels).item()

    @staticmethod
    def _get_batch_num(matrix: Union[np.ndarray, torch.Tensor]) -> int:
        return matrix.shape[0]

    def _average_loss(self, in_loss: float, in_batch_num):
        return (self.avg_loss * self.batch_num + in_loss * in_batch_num) \
            / (self.batch_num + in_batch_num)

    def evaluate(self) -> Dict[str, float]:
        return {MetricKeys.PPL: math.exp(self.avg_loss)}

    def merge(self, other: 'PplMetric'):
        self.avg_loss = self._average_loss(other.avg_loss, other.batch_num)
        self.batch_num += other.batch_num

    def __getstate__(self):
        return self.avg_loss, self.batch_num

    def __setstate__(self, state):
        self.__init__()
        self.avg_loss, self.batch_num = state
