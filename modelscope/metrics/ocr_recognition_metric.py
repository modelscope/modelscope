from typing import Dict

import edit_distance as ed
import numpy as np
import torch
import torch.nn.functional as F

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


def cal_distance(label_list, pre_list):
    y = ed.SequenceMatcher(a=label_list, b=pre_list)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    for item in yy:
        if item[0] == 'insert':
            insert += item[-1] - item[-2]
        if item[0] == 'delete':
            delete += item[2] - item[1]
        if item[0] == 'replace':
            replace += item[-1] - item[-2]
    distance = insert + delete + replace
    return distance, (delete, replace, insert)


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.ocr_recognition_metric)
class OCRRecognitionMetric(Metric):
    """The metric computation class for ocr recognition.
    """

    def __init__(self, *args, **kwargs):
        self.preds = []
        self.targets = []
        self.loss_sum = 0.
        self.nsample = 0
        self.iter_sum = 0

    def add(self, outputs: Dict, inputs: Dict):
        pred = outputs['preds']
        loss = outputs['loss']
        target = inputs['labels']
        self.preds.extend(pred)
        self.targets.extend(target)
        self.loss_sum += loss.data.cpu().numpy()
        self.nsample += len(pred)
        self.iter_sum += 1

    def evaluate(self):
        total_chars = 0
        total_distance = 0
        total_fullmatch = 0
        for (pred, target) in zip(self.preds, self.targets):
            distance, _ = cal_distance(target, pred)
            total_chars += len(target)
            total_distance += distance
            total_fullmatch += (target == pred)
        accuracy = float(total_fullmatch) / self.nsample
        AR = 1 - float(total_distance) / total_chars
        average_loss = self.loss_sum / self.iter_sum if self.iter_sum > 0 else 0
        return {
            MetricKeys.ACCURACY: accuracy,
            MetricKeys.AR: AR,
            MetricKeys.AVERAGE_LOSS: average_loss
        }

    def merge(self, other: 'OCRRecognitionMetric'):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass
