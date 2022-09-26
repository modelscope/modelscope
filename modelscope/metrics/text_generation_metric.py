# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.text_gen_metric)
class TextGenerationMetric(Metric):
    """The metric computation class for text generation classes.

    This metric class calculates F1 of the rouge scores for the whole evaluation dataset.
    """

    def __init__(self):
        self.preds = []
        self.tgts = []
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = outputs['tgts']
        eval_results = outputs['preds']
        self.preds.extend(eval_results)
        self.tgts.extend(ground_truths)

    def evaluate(self):
        scores = [
            self.scorer.score(pred, tgt)['rougeL'].fmeasure
            for pred, tgt in zip(self.preds, self.tgts)
        ]
        return {MetricKeys.F1: sum(scores) / len(scores)}
