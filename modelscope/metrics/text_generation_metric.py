# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Iterable, List

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge import Rouge

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.chinese_utils import rebuild_chinese_str
from modelscope.utils.registry import default_group


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.text_gen_metric)
class TextGenerationMetric(Metric):
    """The metric computation class for text generation classes.

    This metric class calculates F1 of the rouge scores for the whole evaluation dataset.

    Args:
        target_text: The key of the target text column in the `inputs` arg.
        pred_text: The key of the predicted text column in the `outputs` arg.
    """

    def __init__(self, target_text='tgts', pred_text='preds'):
        self.preds: List[str] = []
        self.tgts: List[str] = []
        self.rouge = Rouge()
        self.target_text = target_text
        self.pred_text = pred_text

    def add(self, outputs: Dict[str, List[str]], inputs: Dict[str, List[str]]):
        ground_truths = inputs[self.target_text]
        eval_results = outputs[self.pred_text]
        for truth in ground_truths:
            self.tgts.append(rebuild_chinese_str(truth))
        for result in eval_results:
            self.preds.append(rebuild_chinese_str(result))

    def _check(self, pred: str, tgt: str) -> bool:

        def remove_useless(string: str) -> str:
            return string.replace(' ', '').replace('.', '')

        return len(remove_useless(pred)) != 0 and len(remove_useless(tgt)) != 0

    def evaluate(self):
        assert self.preds, 'preds in TextGenerationMetric must not be empty!'
        tmp = [(pred, tgt) for pred, tgt in zip(self.preds, self.tgts)
               if self._check(pred, tgt)]
        preds, tgts = zip(*tmp)

        def mean(iter: Iterable) -> float:
            return sum(iter) / len(self.preds)

        rouge_scores = self.rouge.get_scores(hyps=preds, refs=tgts)
        rouge_1 = mean(map(lambda score: score['rouge-1']['f'], rouge_scores))
        rouge_l = mean(map(lambda score: score['rouge-l']['f'], rouge_scores))

        pred_list = [each.strip().split(' ') for each in self.preds]
        tgt_list = [[each.strip().split(' ')] for each in self.tgts]
        bleu_1 = corpus_bleu(
            tgt_list,
            pred_list,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method3)
        bleu_4 = corpus_bleu(
            tgt_list,
            pred_list,
            smoothing_function=SmoothingFunction().method3)
        return {
            MetricKeys.ROUGE_1: rouge_1,
            MetricKeys.ROUGE_L: rouge_l,
            MetricKeys.BLEU_1: bleu_1,
            MetricKeys.BLEU_4: bleu_4
        }

    def merge(self, other: 'TextGenerationMetric'):
        self.preds.extend(other.preds)
        self.tgts.extend(other.tgts)

    def __getstate__(self):
        return self.preds, self.tgts

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.tgts = state
