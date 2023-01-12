# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import numpy as np
import torch

from modelscope.metainfo import Metrics
from modelscope.outputs import OutputKeys
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.inbatch_recall)
class InbatchRecallMetric(Metric):
    """The metric computation class for in-batch retrieval classes.

    This metric class calculates in-batch image recall@1 for each input batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inbatch_t2i_hitcnts = []
        self.batch_sizes = []

    def add(self, outputs: Dict, inputs: Dict):
        image_features = outputs[OutputKeys.IMG_EMBEDDING]
        text_features = outputs[OutputKeys.TEXT_EMBEDDING]

        assert type(image_features) == torch.Tensor and type(
            text_features) == torch.Tensor

        with torch.no_grad():
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            batch_size = logits_per_image.shape[0]

            ground_truth = torch.arange(batch_size).long()
            ground_truth = ground_truth.to(image_features.device)

            inbatch_t2i_hitcnt = (logits_per_text.argmax(-1) == ground_truth
                                  ).sum().float().item()

            self.inbatch_t2i_hitcnts.append(inbatch_t2i_hitcnt)
            self.batch_sizes.append(batch_size)

    def evaluate(self):
        assert len(self.inbatch_t2i_hitcnts) == len(
            self.batch_sizes) and len(self.batch_sizes) > 0
        return {
            MetricKeys.BatchAcc:
            sum(self.inbatch_t2i_hitcnts) / sum(self.batch_sizes)
        }

    def merge(self, other: 'InbatchRecallMetric'):
        self.inbatch_t2i_hitcnts.extend(other.inbatch_t2i_hitcnts)
        self.batch_sizes.extend(other.batch_sizes)

    def __getstate__(self):
        return self.inbatch_t2i_hitcnts, self.batch_sizes

    def __setstate__(self, state):
        self.__init__()
        self.inbatch_t2i_hitcnts, self.batch_sizes = state
