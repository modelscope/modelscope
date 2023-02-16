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
    group_key=default_group, module_name=Metrics.loss_metric)
class LossMetric(Metric):
    """The metric class to calculate average loss of batches.

    Args:
        loss_key: The key of loss
    """

    def __init__(self, loss_key=OutputKeys.LOSS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_key = loss_key
        self.losses = []

    def add(self, outputs: Dict, inputs: Dict):
        loss = outputs[self.loss_key]
        self.losses.append(torch_nested_numpify(torch_nested_detach(loss)))

    def evaluate(self):
        return {OutputKeys.LOSS: float(np.average(self.losses))}

    def merge(self, other: 'LossMetric'):
        self.losses.extend(other.losses)

    def __getstate__(self):
        return self.losses

    def __setstate__(self, state):
        self.__init__()
        self.losses = state
