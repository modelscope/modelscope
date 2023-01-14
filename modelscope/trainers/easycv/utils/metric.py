# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
from typing import Dict

import numpy as np
import torch

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS


@METRICS.register_module(module_name='EasyCVMetric')
class EasyCVMetric(Metric):
    """Adapt to ModelScope Metric for EasyCV evaluator.
    """

    def __init__(self, trainer=None, evaluators=None, *args, **kwargs):
        from easycv.core.evaluation.builder import build_evaluator

        self.trainer = trainer
        self.evaluators = build_evaluator(evaluators)
        self.preds = []
        self.grountruths = []

    def add(self, outputs: Dict, inputs: Dict):
        self.preds.append(outputs)
        del inputs

    def evaluate(self):
        results = {}
        for _, batch in enumerate(self.preds):
            for k, v in batch.items():
                if k not in results:
                    results[k] = []
                results[k].append(v)

        for k, v in results.items():
            if len(v) == 0:
                raise ValueError(f'empty result for {k}')

            if isinstance(v[0], torch.Tensor):
                results[k] = torch.cat(v, 0)
            elif isinstance(v[0], (list, np.ndarray)):
                results[k] = list(itertools.chain.from_iterable(v))
            else:
                raise ValueError(
                    f'value of batch prediction dict should only be tensor or list, {k} type is {v[0]}'
                )

        metric_values = self.trainer.eval_dataset.evaluate(
            results, self.evaluators)
        return metric_values

    def merge(self, other: 'EasyCVMetric'):
        self.preds.extend(other.preds)

    def __getstate__(self):
        return self.preds

    def __setstate__(self, state):
        self.__init__()
        self.preds = state
