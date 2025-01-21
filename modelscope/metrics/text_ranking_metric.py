# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.registry import default_group


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.text_ranking_metric)
class TextRankingMetric(Metric):
    """The metric computation class for text ranking classes.

    This metric class calculates mrr and ndcg metric for the whole evaluation dataset.

    Args:
        target_text: The key of the target text column in the `inputs` arg.
        pred_text: The key of the predicted text column in the `outputs` arg.
    """

    def __init__(self, mrr_k: int = 1, ndcg_k: int = 1):
        self.labels: List = []
        self.qids: List = []
        self.logits: List = []
        self.mrr_k: int = mrr_k
        self.ndcg_k: int = ndcg_k

    def add(self, outputs: Dict[str, List], inputs: Dict[str, List]):
        self.labels.extend(inputs.pop('labels').detach().cpu().numpy())
        self.qids.extend(inputs.pop('qid').detach().cpu().numpy())

        logits = outputs['logits'].squeeze(-1).detach().cpu().numpy()
        logits = self._sigmoid(logits).tolist()
        self.logits.extend(logits)

    def evaluate(self):
        rank_result = {}
        for qid, score, label in zip(self.qids, self.logits, self.labels):
            if qid not in rank_result:
                rank_result[qid] = []
            rank_result[qid].append((score, label))

        for qid in rank_result:
            rank_result[qid] = sorted(rank_result[qid], key=lambda x: x[0])

        return {
            MetricKeys.MRR: self._compute_mrr(rank_result),
            MetricKeys.NDCG: self._compute_ndcg(rank_result)
        }

    @staticmethod
    def _sigmoid(logits):
        return np.exp(logits) / (1 + np.exp(logits))

    def _compute_mrr(self, result):
        mrr = 0
        for res in result.values():
            sorted_res = sorted(res, key=lambda x: x[0], reverse=True)
            ar = 0
            for index, ele in enumerate(sorted_res[:self.mrr_k]):
                if str(ele[1]) == '1':
                    ar = 1.0 / (index + 1)
                    break
            mrr += ar
        return mrr / len(result)

    def _compute_ndcg(self, result):
        ndcg = 0
        from sklearn.metrics import ndcg_score
        for res in result.values():
            sorted_res = sorted(res, key=lambda x: [0], reverse=True)
            labels = np.array([[ele[1] for ele in sorted_res]])
            scores = np.array([[ele[0] for ele in sorted_res]])
            ndcg += float(ndcg_score(labels, scores, k=self.ndcg_k))
        return ndcg / len(result)

    def merge(self, other: 'TextRankingMetric'):
        self.labels.extend(other.labels)
        self.qids.extend(other.qids)
        self.logits.extend(other.logits)

    def __getstate__(self):
        return self.labels, self.qids, self.logits, self.mrr_k, self.ndcg_k

    def __setstate__(self, state):
        self.__init__()
        self.labels, self.qids, self.logits, self.mrr_k, self.ndcg_k = state
