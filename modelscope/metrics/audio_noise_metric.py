# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.registry import default_group


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.audio_noise_metric)
class AudioNoiseMetric(Metric):
    """
    The metric computation class for acoustic noise suppression task.
    """

    def __init__(self):
        self.loss = []
        self.amp_loss = []
        self.phase_loss = []
        self.sisnr = []

    def add(self, outputs: Dict, inputs: Dict):
        self.loss.append(outputs['loss'].data.cpu())
        self.amp_loss.append(outputs['amp_loss'].data.cpu())
        self.phase_loss.append(outputs['phase_loss'].data.cpu())
        self.sisnr.append(outputs['sisnr'].data.cpu())

    def evaluate(self):
        avg_loss = sum(self.loss) / len(self.loss)
        avg_sisnr = sum(self.sisnr) / len(self.sisnr)
        avg_amp = sum(self.amp_loss) / len(self.amp_loss)
        avg_phase = sum(self.phase_loss) / len(self.phase_loss)
        total_loss = avg_loss + avg_amp + avg_phase + avg_sisnr
        return {
            'total_loss': total_loss.item(),
            # model use opposite number of sisnr as a calculation shortcut.
            # revert it in evaluation result
            'avg_sisnr': -avg_sisnr.item(),
            MetricKeys.AVERAGE_LOSS: avg_loss.item()
        }

    def merge(self, other: 'AudioNoiseMetric'):
        self.loss.extend(other.loss)
        self.amp_loss.extend(other.amp_loss)
        self.phase_loss.extend(other.phase_loss)
        self.sisnr.extend(other.sisnr)

    def __getstate__(self):
        return self.loss, self.amp_loss, self.phase_loss, self.sisnr

    def __setstate__(self, state):
        self.loss, self.amp_loss, self.phase_loss, self.sisnr = state
