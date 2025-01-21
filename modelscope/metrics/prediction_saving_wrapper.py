# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.prediction_saving_wrapper)
class PredictionSavingWrapper(Metric):
    """The wrapper to save predictions to file.
    Args:
        saving_fn: The saving_fn used to save predictions to files.
    """

    def __init__(self, saving_fn, **kwargs):
        super().__init__(**kwargs)
        self.saving_fn = saving_fn

    def add(self, outputs: Dict, inputs: Dict):
        self.saving_fn(inputs, outputs)

    def evaluate(self):
        return {}

    def merge(self, other: 'PredictionSavingWrapper'):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass
