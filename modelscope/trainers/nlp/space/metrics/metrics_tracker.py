# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from collections import defaultdict


class MetricsTracker(object):
    """ Tracking metrics. """

    def __init__(self):
        self.metrics_val = defaultdict(float)  # for one batch
        self.metrics_avg = defaultdict(float)  # avg batches
        self.num_samples = 0

    def update(self, metrics, num_samples):
        for key, val in metrics.items():
            if val is not None:
                val = float(val)  # [val] -> val
                self.metrics_val[key] = val
                avg_val = \
                    (self.metrics_avg.get(key, 0) * self.num_samples + val * num_samples) / \
                    (self.num_samples + num_samples)
                self.metrics_avg[key] = avg_val
        self.num_samples += num_samples

    def clear(self):
        self.metrics_val = defaultdict(float)
        self.metrics_avg = defaultdict(float)
        self.num_samples = 0

    def items(self):
        return self.metrics_avg.items()

    def get(self, name):
        if self.num_samples == 0:
            raise ValueError('There is no data in Metrics.')
        return self.metrics_avg.get(name)

    def state_dict(self):
        return {
            'metrics_val': self.metrics_val,
            'metrics_avg': self.metrics_avg,
            'num_samples': self.num_samples,
        }

    def load_state_dict(self, state_dict):
        self.metrics_val = state_dict['metrics_val']
        self.metrics_avg = state_dict['metrics_avg']
        self.num_samples = state_dict['num_samples']

    def value(self):
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = f'{key.upper()}-{val:.3f}'
            metric_strs.append(metric_str)
        if 'token_nll' in self.metrics_val:
            metric_str = f"TOKEN_PPL-{math.exp(self.metrics_val['token_nll']):.3f}"
            metric_strs.append(metric_str)
        metric_strs = '   '.join(metric_strs)
        return metric_strs

    def summary(self):
        metric_strs = []
        for key, val in self.metrics_avg.items():
            metric_str = f'{key.upper()}-{val:.3f}'
            metric_strs.append(metric_str)
        if 'token_nll' in self.metrics_avg:
            metric_str = f"TOKEN_PPL-{math.exp(self.metrics_avg['token_nll']):.3f}"
            metric_strs.append(metric_str)
        metric_strs = '   '.join(metric_strs)
        return metric_strs
